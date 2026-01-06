import sys
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor, ContrastivePreProcessor
from ..arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
CONTRASTIVE_PROCESSORS = [ContrastivePreProcessor, QueryPreProcessor, CorpusPreProcessor]

PROCESSOR_INFO = {
    "json": [None, None, None],
    "Tevatron/wikipedia-nq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-trivia": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-curated": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-wq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-squad": DEFAULT_PROCESSORS,
    "Tevatron/scifact": DEFAULT_PROCESSORS,
    "Tevatron/beir": DEFAULT_PROCESSORS,
    "Tevatron/msmarco-passage": DEFAULT_PROCESSORS,
    "Tevatron/msmarco-passage-corpus": DEFAULT_PROCESSORS,
    "Tevatron/msmarco-passage-aug": DEFAULT_PROCESSORS,
    "brutusxu/nomic-embed-pretrain-lite": CONTRASTIVE_PROCESSORS,
}

# Get HuggingFace token from environment variable
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


class HFTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_language,
            data_files=data_files,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True,
        )[data_args.dataset_split]

        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][0]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[0]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )
        self.lowercase = data_args.lowercase
        self.add_eos_token = data_args.add_eos_token

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator, lowercase=self.lowercase, add_eos_token=self.add_eos_token),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False,  # disable cache file to avoid cache file conflict
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_language,
            data_files=data_files,
            token=hf_token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )[data_args.dataset_split]
        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][1]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[1]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num
        self.lowercase = data_args.lowercase
        self.add_eos_token = data_args.add_eos_token

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, lowercase=self.lowercase, add_eos_token=self.add_eos_token),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
                load_from_cache_file=False,  # disable cache file to avoid cache file conflict
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_language,
            data_files=data_files,
            token=hf_token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )[data_args.dataset_split]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith("-corpus"):
            script_prefix = script_prefix[:-7]
        self.preprocessor = (
            PROCESSOR_INFO[script_prefix][2] if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        )
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )
        self.lowercase = data_args.lowercase
        self.add_eos_token = data_args.add_eos_token
        # currently hardcoded, need to change in the future
        if data_args.dataset_name == "Tevatron/beir-corpus" and (data_args.dataset_language == "quora" or "cqadupstack" in data_args.dataset_language):
            self.p_prefix = "query: "
        else:
            self.p_prefix = "passage: "

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator, self.p_prefix, lowercase=self.lowercase, add_eos_token=self.add_eos_token),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
                load_from_cache_file=False,  # disable cache file to avoid cache file conflict
            )
        return self.dataset
