import os
import sys
import json
from typing import Dict
import torch
import logging
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM

from huggingface_hub import snapshot_download
from filelock import FileLock
import tempfile

from llm2vec.models import LlamaBiForMNTP
from llm2vec.models import Qwen2BiForMNTP
from .encoder import EncoderModel
from .encoder import DecoderModel
from .encoder import EncoderOutput

logger = logging.getLogger(__name__)


class SpladeModel(EncoderModel):
    TRANSFORMER_CLS = AutoModelForMaskedLM

    def encode_passage(self, psg):
        """
        psg["input_ids"]: torch.Tensor (bz, seq_len)
        psg["attention_mask"]: torch.Tensor (bz, seq_len)
        """
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(
            torch.log(1 + torch.relu(psg_out)) * psg["attention_mask"].unsqueeze(-1),
            dim=1,
        )
        return aggregated_psg_out

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(
            torch.log(1 + torch.relu(qry_out)) * qry["attention_mask"].unsqueeze(-1),
            dim=1,
        )
        return aggregated_psg_out

    def _apply_hard_topk(self, representations, k):
        """
        Apply hard top-k sparsity constraint to representations.
        
        Args:
            representations: (batch_size, vocab_size)
            k: number of terms to keep
        
        Returns:
            Sparse representations with at most k non-zero values
        """
        if self.training:
            # During training: use straight-through estimator for differentiability
            topk_values, topk_indices = torch.topk(representations, k, dim=-1)
            threshold = topk_values[:, -1:].detach()  # Detach for straight-through gradient
            
            # Create mask: 1 for values >= threshold, 0 otherwise
            mask = (representations >= threshold).float()
            
            # Apply mask with straight-through gradient
            sparse_reps = representations * mask
            
            return sparse_reps
        else:
            # During inference: hard top-k
            batch_size, vocab_size = representations.shape
            sparse_reps = torch.zeros_like(representations)
            topk_values, topk_indices = torch.topk(representations, k, dim=-1)
            
            # Scatter top-k values back
            sparse_reps.scatter_(1, topk_indices, topk_values)
            
            return sparse_reps



class SpladeModelForCausalLM(DecoderModel):
    # TRANSFORMER_CLS is set dynamically based on model name in build() and load() methods
    TRANSFORMER_CLS = None

    @classmethod
    def get_transformer_cls(cls, model_name_or_path: str, is_bidirectional: bool=False):
        """
        Determine the appropriate transformer class based on the model name/path.
        
        Args:
            model_name_or_path: The model name or path to determine the transformer class
            
        Returns:
            The appropriate transformer class (LlamaBiForMNTP, Qwen2BiForMNTP, or AutoModelForCausalLM)
        """
        model_name_lower = model_name_or_path.lower()
        
        if ("llama" in model_name_lower or "laconic" in model_name_lower) and is_bidirectional:
            return LlamaBiForMNTP
        elif "qwen" in model_name_lower and is_bidirectional:
            return Qwen2BiForMNTP
        else:
            # Default to AutoModelForCausalLM for other models
            return AutoModelForCausalLM

    def _apply_hard_topk(self, representations, k):
        """
        Apply hard top-k sparsity constraint to representations.
        
        Args:
            representations: (batch_size, vocab_size)
            k: number of terms to keep
        
        Returns:
            Sparse representations with at most k non-zero values
        """
        if self.training:
            # During training: use straight-through estimator for differentiability
            topk_values, topk_indices = torch.topk(representations, k, dim=-1)
            threshold = topk_values[:, -1:].detach()  # Detach for straight-through gradient
            
            # Create mask: 1 for values >= threshold, 0 otherwise
            mask = (representations >= threshold).float()
            
            # Apply mask with straight-through gradient
            sparse_reps = representations * mask
            
            return sparse_reps
        else:
            # During inference: hard top-k
            batch_size, vocab_size = representations.shape
            sparse_reps = torch.zeros_like(representations)
            topk_values, topk_indices = torch.topk(representations, k, dim=-1)
            
            # Scatter top-k values back
            sparse_reps.scatter_(1, topk_indices, topk_values)
            
            return sparse_reps
    
    def _aggregate_sequence(self, logits, attention_mask):
        """
        Aggregate logits over sequence dimension using specified pooling strategy.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            aggregated: (batch_size, vocab_size)
        """
        if self.pooling_strategy == "max":
            # Current max-pooling implementation
            aggregated = torch.log(
                torch.relu(torch.max(logits + (1 - attention_mask.unsqueeze(-1)) * -1e6, dim=1)[0]) + 1
            )
        
        elif self.pooling_strategy == "last":
            # Last token pooling - use attention_mask to find last real token
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,) zero-indexed
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            last_token_logits = logits[batch_indices, seq_lengths, :]  # (batch_size, vocab_size)
            aggregated = torch.log(torch.relu(last_token_logits) + 1)
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padding tokens
            masked_logits = logits * attention_mask.unsqueeze(-1)
            sum_logits = masked_logits.sum(dim=1)  # (batch_size, vocab_size)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
            mean_logits = sum_logits / seq_lengths.clamp(min=1)
            aggregated = torch.log(torch.relu(mean_logits) + 1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return aggregated

    def encode_passage(self, psg):
        """
        psg["input_ids"]: torch.Tensor (bz, seq_len)
        psg["attention_mask"]: torch.Tensor (bz, seq_len)
        """

        if psg is None:
            return None
        
        if self.is_echo:
            input_ids = psg["input_ids"].repeat(1, 2)  # (bz, seq_len * 2)
            attention_mask = psg["attention_mask"].repeat(1, 2)  # (bz, seq_len * 2)
            psg_out = self.lm_p(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits
            psg_out = psg_out[:, psg_out.size(1) // 2 :, :]
            attention_mask = attention_mask[:, attention_mask.size(1) // 2 :]
        else:
            psg_out = self.lm_p(**psg, return_dict=True).logits
            attention_mask = psg["attention_mask"]
        
        # Use the new aggregation method with pooling strategy
        aggregated_psg_out = self._aggregate_sequence(psg_out, attention_mask)

        # Apply hard top-k constraint if p_max_terms is set
        if self.p_max_terms is not None:
            aggregated_psg_out = self._apply_hard_topk(aggregated_psg_out, self.p_max_terms)

        return aggregated_psg_out

    def encode_query(self, qry):
        """
        qry["input_ids"]: torch.Tensor (bz, seq_len)
        qry["attention_mask"]: torch.Tensor (bz, seq_len)
        """
        if qry is None:
            return None
        
        if self.is_echo:
            input_ids = qry["input_ids"].repeat(1, 2)  # (bz, seq_len * 2)
            attention_mask = qry["attention_mask"].repeat(1, 2)  # (bz, seq_len * 2)
            qry_out = self.lm_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits  # (bz, seq_len * 2, vocab_size)
            qry_out = qry_out[:, qry_out.size(1) // 2 :, :]
            attention_mask = attention_mask[:, attention_mask.size(1) // 2 :]
        else:
            qry_out = self.lm_q(**qry, return_dict=True).logits
            attention_mask = qry["attention_mask"]
        
        # Use the new aggregation method with pooling strategy
        aggregated_qry_out = self._aggregate_sequence(qry_out, attention_mask)

        # Apply hard top-k constraint if q_max_terms is set
        if self.q_max_terms is not None:
            aggregated_qry_out = self._apply_hard_topk(aggregated_qry_out, self.q_max_terms)

        return aggregated_qry_out

    @classmethod
    def build(cls, model_args, train_args, **hf_kwargs):
        """
        Override build method to dynamically set TRANSFORMER_CLS based on model name
        """
        # Dynamically set TRANSFORMER_CLS based on model name
        cls.TRANSFORMER_CLS = cls.get_transformer_cls(model_args.model_name_or_path, train_args.is_bidirectional)
        logger.info(f"Using transformer class: {cls.TRANSFORMER_CLS.__name__}")
        
        # Call parent build method which will use the dynamically set TRANSFORMER_CLS
        return super().build(model_args, train_args, **hf_kwargs)

    @classmethod
    def load(
        cls,
        model_name_or_path,
        normalize: bool = False,
        is_echo: bool = False,
        is_bidirectional: bool = False,
        load_type="auto", 
        pooling_strategy="max",
        local_files_only: bool = False,
        **hf_kwargs,
    ):
        """
        this function overwrites the load function in EncoderModel, if we want to load a model trained with echo, is_echo needs to be set to True
        """
        # Add local_files_only to hf_kwargs
        if local_files_only:
            hf_kwargs['local_files_only'] = True
        
        # Dynamically set TRANSFORMER_CLS based on model name
        cls.TRANSFORMER_CLS = cls.get_transformer_cls(model_name_or_path, is_bidirectional=is_bidirectional)
        logger.info(f"Using transformer class: {cls.TRANSFORMER_CLS.__name__}")
        
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, "query_model")
            _psg_model_path = os.path.join(model_name_or_path, "passage_model")
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                if load_type == "flash_attn":
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                        _qry_model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        **hf_kwargs,
                    )
                else:
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                if load_type == "flash_attn":
                    lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                        _psg_model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        **hf_kwargs,
                    )
                else:
                    lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(f"loading model weight from {model_name_or_path}")
                if load_type == "flash_attn":
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        **hf_kwargs,
                    )
                elif load_type == "int8":
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                        model_name_or_path,
                        load_in_8bit=True,
                        **hf_kwargs
                    )
                elif load_type == "int4":
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                        model_name_or_path,
                        load_in_4bit=True,
                        **hf_kwargs
                    )
                elif load_type == "gptq":
                    from auto_gptq import AutoGPTQForCausalLM
                    lm_q = AutoGPTQForCausalLM.from_quantized(
                        model_name_or_path
                    )
                else:
                    lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                        model_name_or_path, 
                        **hf_kwargs,
                        )
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
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
            pooling_strategy=pooling_strategy,
        )
        return model
