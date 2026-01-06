import copy
import json
import os
import time
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM
from transformers.file_utils import ModelOutput
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from huggingface_hub import snapshot_download
from filelock import FileLock
import tempfile


from tevatron.arguments import (
    ModelArguments,
    TevatronTrainingArguments as TrainingArguments,
)

import logging

logger = logging.getLogger(__name__)

# Get HuggingFace token from environment variable
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


def ensure_model_downloaded(model_name_or_path: str, token: str = None, cache_dir: str = None):
    """
    Pre-download model files in a process-safe manner.
    Should be called in the main process before spawning workers.
    
    Args:
        model_name_or_path: Model identifier or path
        token: HuggingFace token if needed
        cache_dir: Custom cache directory
        
    Returns:
        str: Path to the downloaded model
    """
    # Skip if it's already a local directory
    if os.path.isdir(model_name_or_path):
        logger.info(f"Model is already local at {model_name_or_path}")
        return model_name_or_path
    
    # Create a lock file in temp directory to coordinate across processes
    lock_dir = cache_dir if cache_dir else os.path.join(tempfile.gettempdir(), 'huggingface_locks')
    os.makedirs(lock_dir, exist_ok=True)
    
    # Sanitize model name for lock file
    lock_file_name = model_name_or_path.replace('/', '_').replace('\\', '_') + '.lock'
    lock_path = os.path.join(lock_dir, lock_file_name)
    
    with FileLock(lock_path, timeout=3600):  # 1 hour timeout
        logger.info(f"Downloading model {model_name_or_path} (process-safe)")
        try:
            model_path = snapshot_download(
                repo_id=model_name_or_path,
                token=token or hf_token,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
            )
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    """
    this pooler class is not used in training SPLADE
    """

    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError("EncoderPooler is an abstract class")

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, "pooler.pt")
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f"Loading Pooler from {pooler_path}")
                state_dict = torch.load(pooler_path, map_location="cpu")
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "pooler.pt"))
        with open(os.path.join(save_path, "pooler_config.json"), "w") as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        pooler: nn.Module = None,
        untie_encoder: bool = False,
        temperature: float = 1.0,
        normalize: bool = False,
        is_echo: bool = False,
        negatives_x_device: bool = False,
        q_max_terms: Optional[int] = None,
        p_max_terms: Optional[int] = None,
        pooling_strategy: str = "max",
    ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        self.temperature = temperature
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError("Distributed training has not been initialized for representation all gather.")
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.normalize = normalize
        self.is_echo = is_echo
        self.q_max_terms = q_max_terms
        self.p_max_terms = p_max_terms
        self.pooling_strategy = pooling_strategy


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        """
        # TODO: does this stabilize the training?
        q_reps *= self.lm_q.base_model.config.hidden_size**-0.25
        p_reps *= self.lm_p.base_model.config.hidden_size**-0.25  
        """
        if self.normalize:  # here we apply the normalization
            p_reps = nn.functional.normalize(p_reps, p=2, dim=-1)
            q_reps = nn.functional.normalize(q_reps, p=2, dim=-1)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # here we apply the temperature
            scores = scores.view(q_reps.size(0), -1)  # (bz, num_items)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            # print(scores.shape, target.shape)

            loss = self.compute_loss(scores, target)

            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    # def min_max_normalize(self, scores: torch.Tensor):
    #     min_val, _ = torch.min(scores, dim=1)
    #     max_val, _ = torch.max(scores, dim=1)
    #     normalized_scores = (scores - min_val) / (max_val - min_val)
    #     return normalized_scores

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError("EncoderModel is an abstract class")

    def encode_query(self, qry):
        raise NotImplementedError("EncoderModel is an abstract class")

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        local_files_only: bool = False,
        **hf_kwargs,
    ):
        """
        here the config is being used
        """

        # Add local_files_only to hf_kwargs if not already present
        if local_files_only:
            hf_kwargs['local_files_only'] = True

        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, "query_model")
                _psg_model_path = os.path.join(model_args.model_name_or_path, "passage_model")
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, token=hf_token, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            temperature=train_args.temperature,
            untie_encoder=model_args.untie_encoder,
            normalize=model_args.normalize,
            is_echo=train_args.is_echo,
            q_max_terms=train_args.q_max_terms if train_args.use_hard_topk else None,
            p_max_terms=train_args.p_max_terms if train_args.use_hard_topk else None,
            pooling_strategy=train_args.pooling_strategy,
        )
        # model.model_accepts_loss_kwargs = True
        return model

    @classmethod
    def load(
        cls,
        model_name_or_path,
        normalize: bool = False,
        is_echo: bool = False,
        load_type="auto",
        local_files_only: bool = False,
        pooling_strategy="max",
        **hf_kwargs,
    ):
        """
        as load function is shared by both encoder and decoder model, we need to pass in flags like normalize and is_echo
        """
        # Add local_files_only to hf_kwargs if not already present
        if local_files_only:
            hf_kwargs['local_files_only'] = True

        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, "query_model")
            _psg_model_path = os.path.join(model_name_or_path, "passage_model")
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(f"loading model weight from {model_name_or_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, 
                token=hf_token,
                **hf_kwargs
                )
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, "pooler.pt")
        pooler_config = os.path.join(model_name_or_path, "pooler_config.json")
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f"found pooler weight and configuration")
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
            normalize=normalize,
            is_echo=is_echo,
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            self.lm_q.save_pretrained(
                output_dir, safe_serialization=False
            )  # note, as mosaic bert has a weight sharing, this needs to be further examined
        if self.pooler:
            self.pooler.save_pooler(output_dir)


class DecoderModel(EncoderModel):
    TRANSFORMER_CLS = AutoModel

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        local_files_only: bool = False,
        **hf_kwargs,
    ):
        # Add local_files_only to hf_kwargs
        if local_files_only:
            hf_kwargs['local_files_only'] = True

        attn_implementation = "flash_attention_2" if train_args.use_flash_attn else "sdpa"  # default to sdpa
        torch_dtype = torch.bfloat16 if train_args.use_flash_attn else torch.float32
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, "query_model")
                _psg_model_path = os.path.join(model_args.model_name_or_path, "passage_model")
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation, **hf_kwargs
                )
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    **hf_kwargs,
                )
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path,
                    token=hf_token,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    **hf_kwargs,
                )
                lm_p = lm_q

        # load pre-trained
        else:
            if model_args.qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )

                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path,
                    quantization_config=bnb_config,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                )
                lm_q = prepare_model_for_kbit_training(lm_q)
                lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    **hf_kwargs,
                )
                lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if train_args.gradient_checkpointing:
            lm_q.gradient_checkpointing_enable()
            train_args.gradient_checkpointing = False  # we change it to False to avoid trainer error

        if model_args.lora:
            # target_modules = "all-linear"
            if "qwen2.5" in model_args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            elif "llama" in model_args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "qwen3" in model_args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            else:
                raise ValueError("Please specify target modules for LoRA adaptation.")

            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            lm_q = get_peft_model(lm_q, lora_config)
            # check untie_encoder
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q
            lm_q.print_trainable_parameters()

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            temperature=train_args.temperature,
            untie_encoder=model_args.untie_encoder,
            normalize=model_args.normalize,
            is_echo=train_args.is_echo,
            q_max_terms=train_args.q_max_terms if train_args.use_hard_topk else None,
            p_max_terms=train_args.p_max_terms if train_args.use_hard_topk else None,
            pooling_strategy=train_args.pooling_strategy,
        )
        # model.model_accepts_loss_kwargs = True

        return model

    def save(self, output_dir: str):
        # logger.info("This function is being called")
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            # TODO: understand whether this causes the frozen problem
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)
