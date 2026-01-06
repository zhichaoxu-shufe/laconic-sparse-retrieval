import logging
import os
import sys
import json
from typing import Optional
from packaging import version

import torch
import torch.distributed as dist

from transformers import PreTrainedModel
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from peft import PeftModel
from accelerate import __version__ as accelerate_version
from dataclasses import dataclass, field

import wandb

from utils import replace_with_xformers_attention

from tevatron.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.data import TrainDataset, QPCollator
from tevatron.modeling import SpladeModelForCausalLM
from tevatron.trainer import TevatronTrainer
from tevatron.datasets import HFTrainDataset
from tevatron.modeling.encoder import ensure_model_downloaded

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

logger = logging.getLogger(__name__)

# Get HuggingFace token from environment variable
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
if not hf_token:
    logger.warning("No HuggingFace token found in environment variables (HF_TOKEN or HUGGING_FACE_HUB_TOKEN). Some models may not be accessible.")


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T"""
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


@dataclass
class SpladeTrainingArguments(TevatronTrainingArguments):
    # inherit from TevatronTrainingArguments, with additional FLOPS regularization arguments
    q_flops_loss_factor: float = field(default=1e-2)
    p_flops_loss_factor: float = field(default=1e-2)
    flops_warmup: int = field(default=1000, metadata={"help": "number of warmup steps"})
    use_flops_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use FLOPS regularization loss. Set to False to disable entirely."}
    )


class SpladeTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super(SpladeTrainer, self).__init__(*args, **kwargs)
        if self.args.negatives_x_device:
            self.world_size = torch.distributed.get_world_size()

        self.q_flops_scheduler = RegWeightScheduler(self.args.q_flops_loss_factor, self.args.flops_warmup)
        self.p_flops_scheduler = RegWeightScheduler(self.args.p_flops_loss_factor, self.args.flops_warmup)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        TODO: investigate why save pretrained model is not working with FSDP
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                self.model.save(output_dir)
                logger.info(f"checkpoint saved {output_dir}")

        elif self.args.should_save:
            self._save(output_dir)

    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def compute_loss(self, model, inputs):
        query, passage = inputs
        output = model(query=query, passage=passage)

        q_reps = output.q_reps
        p_reps = output.p_reps

        # Calculate mean activated dimensions (non-zero elements) per minibatch
        q_activated_dims = (q_reps > 0).sum(dim=-1).float().mean().item()
        p_activated_dims = (p_reps > 0).sum(dim=-1).float().mean().item()

        # Standard single-dimension training, conditionally compute FLOPS loss
        if self.args.use_flops_loss:
            q_flops_loss = self.q_flops_scheduler.get_lambda() * self._flops(q_reps)
            p_flops_loss = self.p_flops_scheduler.get_lambda() * self._flops(p_reps)
            if self.args.negatives_x_device:
                q_flops_loss *= self.world_size
                p_flops_loss *= self.world_size
        else:
            q_flops_loss = 0.0
            p_flops_loss = 0.0

        if model.normalize:
            p_reps = torch.nn.functional.normalize(p_reps, p=2, dim=-1)
            q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=-1)
        
        scores = model.compute_similarity(q_reps, p_reps) / model.temperature
        scores = scores.view(q_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss = model.cross_entropy(scores, target)

        if self.args.negatives_x_device:
            loss = loss * self.world_size
        
        total_loss = loss + q_flops_loss + p_flops_loss
        
        # Log to wandb
        if self.is_world_process_zero() and wandb.run is not None:
            wandb.log({
                'train/total_loss': total_loss.item(),
                'train/main_loss': loss.item(),
                'train/q_flops_loss': q_flops_loss.item() if isinstance(q_flops_loss, torch.Tensor) else q_flops_loss,
                'train/p_flops_loss': p_flops_loss.item() if isinstance(p_flops_loss, torch.Tensor) else p_flops_loss,
                'train/q_flops_lambda': self.q_flops_scheduler.get_lambda(),
                'train/p_flops_lambda': self.p_flops_scheduler.get_lambda(),
                'train/q_mean_activated_dims': q_activated_dims,
                'train/p_mean_activated_dims': p_activated_dims,
            }, step=self.state.global_step)
            
        return total_loss

    def training_step(self, *args):
        self.q_flops_scheduler.step()
        self.p_flops_scheduler.step()
        return super(SpladeTrainer, self).training_step(*args)


TrainingArguments = SpladeTrainingArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    model_args.model_name_or_path = model_args.model_name_or_path

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    # Initialize wandb only on global rank 0
    if dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = training_args.local_rank in [-1, 0]

    if is_main_process:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'splade-training'),
                name=os.environ.get('WANDB_RUN_NAME', training_args.run_name or training_args.output_dir.split('/')[-1]),
                config={
                    **vars(training_args),
                    **vars(model_args),
                    **vars(data_args),
                }
            )
            logger.info("Wandb initialized successfully")
        else:
            logger.warning("WANDB_API_KEY not found in environment variables. Wandb logging disabled.")

    num_labels = 1
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels,
        token=hf_token,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    model_args.tokenizer_name = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        token=hf_token,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    # this is aligned with encode_splade_causal.py
    if not tokenizer.pad_token or not tokenizer.pad_token_id:
        tokenizer.pad_token, tokenizer.pad_token_id = (
            tokenizer.eos_token,
            tokenizer.eos_token_id,
        )
    # by default use right padding and truncation, as https://arxiv.org/pdf/2207.01262 suggests the bias of the current benchmarks
    tokenizer.padding_side, tokenizer.truncation_side = "right", "right"

    # Pre-download model on main process to avoid race conditions
    # This must be done before distributed initialization or model building
    if training_args.local_rank in [-1, 0]:  # Main process only
        logger.info("Pre-downloading model to ensure process-safe loading...")
        ensure_model_downloaded(
            model_name_or_path=model_args.model_name_or_path,
            token=hf_token,
            cache_dir=model_args.cache_dir
        )
        logger.info("Model pre-download complete")
    
    # Synchronization barrier if using distributed training
    if dist.is_initialized():
        logger.info("Waiting for all processes to reach barrier after model download...")
        dist.barrier()
        logger.info("All processes synchronized, proceeding with model loading")

    model = SpladeModelForCausalLM.build(
        model_args,
        training_args,
        config=config,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        local_files_only=True,  # Ensure loading from local files only
    )
    model.prepare_inputs_for_generation = None

    train_dataset = HFTrainDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )

    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)
    trainer = SpladeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
