import logging
import os
import time
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import json

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import (
    ModelArguments,
    DataArguments,
    TevatronTrainingArguments as TrainingArguments,
)
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.modeling import EncoderOutput, SpladeModel
from tevatron.modeling import SpladeModelForCausalLM
from tevatron.modeling.encoder import ensure_model_downloaded
from tevatron.datasets import HFQueryDataset, HFCorpusDataset

from models import MODELS_DICT

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    model_args.tokenizer_name = (
        model_args.tokenizer_name if model_args.tokenizer_name != "" else model_args.model_name_or_path
    )

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    if not tokenizer.pad_token or not tokenizer.pad_token_id:
        tokenizer.pad_token, tokenizer.pad_token_id = (
            tokenizer.eos_token,
            tokenizer.eos_token_id,
        )
    tokenizer.padding_side, tokenizer.truncation_side = "right", "right"

    # Pre-download model to avoid race conditions in concurrent processes
    if training_args.local_rank <= 0:  # Only pre-download on main process
        logger.info("Pre-downloading model to ensure process-safe loading...")
        ensure_model_downloaded(
            model_name_or_path=model_args.model_name_or_path,
            token=None,  # Token is auto-detected from environment
            cache_dir=model_args.cache_dir
        )
        logger.info("Model pre-download complete")

    # TODO: investigate the problem in model loading with config specified
    model = SpladeModelForCausalLM.load(
        model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        normalize=model_args.normalize,
        is_echo=training_args.is_echo,
        is_bidirectional=training_args.is_bidirectional,
        load_type=model_args.load_type,
        pooling_strategy=training_args.pooling_strategy,
        local_files_only=True,  # Ensure loading from local files only
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            # cache_dir=data_args.data_cache_dir or model_args.cache_dir,
            cache_dir=None,
        )
    else:
        encode_dataset = HFCorpusDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            # cache_dir=data_args.data_cache_dir or model_args.cache_dir,
            cache_dir=None,
        )
    encode_dataset = EncodeDataset(
        encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
        tokenizer,
        max_len=text_max_length,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(tokenizer, max_length=text_max_length, padding="max_length"),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    lookup_indices = []
    model = model.to(training_args.device)
    if (training_args.bf16 or training_args.fp16):
        model = model.to(torch.float16) 
    # model = torch.compile(model)
    model.eval()
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    collection_file = open(data_args.encoded_save_path, "w")

    pbar_step = 100
    pbar = tqdm(encode_loader)
    for batch_idx, (batch_ids, batch) in enumerate(encode_loader):
        if batch_idx % pbar_step == 0:
            pbar.update(pbar_step)
        
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():  # if bf16, what will happen?
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch)
                    reps = model_output.q_reps.cpu().detach()  # torch.tensor
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    reps = model_output.p_reps.cpu().detach()  # torch.tensor
                
                for rep, id_ in zip(reps, batch_ids):
                    # Determine top-k dimension
                    if data_args.encode_is_qry:
                        topk_k = training_args.q_max_terms
                    else:
                        topk_k = training_args.p_max_terms
                    
                    topk_values, topk_indices = torch.topk(rep, topk_k)
                    topk_values, topk_indices, rep = topk_values.numpy(), topk_indices.numpy(), rep.numpy()
                    data = rep[topk_indices]
                    
                    # Choose encoding format based on use_float_weights flag
                    if training_args.use_float_weights:
                        # Float precision - keep original values
                        dict_splade = dict()
                        for id_token, value_token in zip(topk_indices, data):
                            if value_token > 0:
                                real_token = vocab_dict.get(id_token, None)
                                if real_token is not None:
                                    dict_splade[real_token] = float(value_token)
                    else:
                        # Integer quantization (for Lucene/Anserini compatibility)
                        data = np.rint(data * 100).astype(int)
                        dict_splade = dict()
                        for id_token, value_token in zip(topk_indices, data):
                            if value_token > 0:
                                real_token = vocab_dict.get(id_token, None)  # in the case of qwen3, model vocab is larger than tokenizer vocab
                                if real_token is not None:
                                    dict_splade[real_token] = int(value_token)
                    
                    if len(dict_splade.keys()) == 0:
                        print("empty input =>", id_)
                        # Use appropriate fallback value based on format
                        fallback_value = 1.0 if training_args.use_float_weights else 1
                        dict_splade[tokenizer.pad_token] = fallback_value
                    
                    if not data_args.encode_is_qry:
                        # Corpus: always use JSON format
                        dict_ = dict(id=id_, content="", vector=dict_splade)
                        json_dict = json.dumps(dict_)
                        collection_file.write(json_dict + "\n")
                    else:
                        # Queries: format depends on use_float_weights
                        if training_args.use_float_weights:
                            # JSON dict format for float weights
                            json_dict = json.dumps(dict_splade)
                            collection_file.write(str(id_) + "\t" + json_dict + "\n")
                        else:
                            # Repeated tokens format for integer weights (Lucene compatibility)
                            string_splade = " ".join(
                                [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()]
                            )
                            collection_file.write(str(id_) + "\t" + string_splade + "\n")
    collection_file.close()
    pbar.close()

if __name__ == "__main__":
    main()
