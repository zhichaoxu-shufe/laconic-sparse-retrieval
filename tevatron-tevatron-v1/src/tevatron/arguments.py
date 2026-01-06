import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"},
    )

    # lora
    lora: bool = field(default=False, metadata={"help": "whether to use lora"})
    qlora: bool = field(default=False, metadata={"help": "whether to use qlora"})
    lora_r: int = field(default=128, metadata={"help": "rank of lora"})
    lora_alpha: int = field(default=256, metadata={"help": "alpha of lora"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
            "of `[float32, float16, bfloat16]`. "
        },
    )

    load_type: str = field(default="auto", metadata={"help": "load type for inference optimization"})


@dataclass
class DataArguments:
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    passage_field_separator: str = field(default=" ")
    dataset_proc_num: int = field(default=12, metadata={"help": "number of proc used in dataset preprocess"})
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the data downloaded from huggingface"},
    )
    
    lowercase: bool = field(
        default=False,
        metadata={"help": "Whether to lowercase all text before tokenization (recommended for BPE tokenizers to avoid case-sensitive token fragmentation)"}
    )
    
    add_eos_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to append EOS token to sequences. "
                    "Recommended when using pooling_strategy='last' for decoder-only models. "
                    "Default is False to maintain current behavior."
        }
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split("/")
            self.dataset_split = info[-1] if len(info) == 3 else "train"
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else "/".join(info)
            self.dataset_language = "default"
            if ":" in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(":")
        else:
            self.dataset_name = "json"
            self.dataset_split = "train"
            self.dataset_language = "default"
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                # change all train directory paths to absolute
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    learning_rate: float = field(default=0.1)
    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={"help": "choose from following options: constant_with_warmup, linear, cosine"},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "whether to do sequence packing, used in SFTTrainer"},
    )

    bf16: bool = field(default=False, metadata={"help": "whether to use bf16"})
    fp16: bool = field(default=False, metadata={"help": "whether to use fp16"})
    use_flash_attn: bool = field(default=False, metadata={"help": "use flash attention"})

    temperature: float = field(default=1.0, metadata={"help": "temperature for softmax"})
    is_echo: bool = field(default=False, metadata={"help": "whether to use echo dataset"})
    is_bidirectional: bool = field(default=False, metadata={"help": "whether to use bidirectional attention for the causal model"})

    optim: str = field(default="adamw_torch", metadata={"help": "optimizer of choice"})
    warmup_ratio: float = field(default=0.1)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=False, metadata={"help": "use gradient checkpointing"})
    torch_compile: bool = field(default=False, metadata={"help": "whether to use torch compile"})

    save_only_model: bool = field(
        default=True,
        metadata={"help": "save only model or save optimizer state and scheduler state as well"},
    )

    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=4)

    run_name: Optional[str] = field(
        default=None,
    )
    report_to: Optional[List[str]] = field(
        default="none",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )

    # top-k sparsity constraints, these arguments are also used in encoding stage
    use_hard_topk: bool = field(
        default=False,
        metadata={"help": "Whether to use hard top-k sparsity (only keep top-k activations) during training."}
    )

    q_max_terms: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of non-zero terms for query representation (e.g., 128, 256). Set to None to disable."}
    )
    p_max_terms: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of non-zero terms for passage representation (e.g., 256, 512). Set to None to disable."}
    )

    # Float weights for better precision
    use_float_weights: bool = field(
        default=True,
        metadata={"help": "Save weights as floats instead of quantized integers. Set to False for Lucene/Anserini compatibility (will use integer quantization: value * 100)."}
    )
    
    # Pooling strategy for sequence aggregation
    pooling_strategy: str = field(
        default="max",
        metadata={
            "help": "Pooling strategy for sequence aggregation. Options: 'max' (max-pool over sequence), "
                    "'last' (use last token only - recommended for decoder-only models), 'mean' (average over sequence). "
                    "Default is 'max' to maintain current behavior."
        }
    )
