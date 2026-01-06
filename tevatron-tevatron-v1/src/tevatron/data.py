import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=(self.data_args.q_max_len if is_query else self.data_args.p_max_len),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        current understanding is the positive passage will always be first
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group["positives"]
        group_negatives = group["negatives"]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset : _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages


class DistilTrainDataset(TrainDataset):
    def __init__(
        self,
        data_args: DataArguments,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        trainer: TevatronTrainer = None,
    ):
        super().__init__(data_args, dataset, tokenizer, trainer)

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], List[float], List[float]]:
        """
        current understanding is the positive passage will always be first
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        positive_scores = []
        negative_scores = []
        group_positives = group["positives"]
        group_negative = group["negatives"]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            pos_score = group["positive_scores"][0]
        else:
            idx = (_hashed_seed + epoch) % len(group_positives)
            pos_psg = group_positives[idx]
            pos_score = group["positive_scores"][idx]
        encoded_passages.append(self.create_one_example(pos_psg))
        positive_scores.append(pos_score)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negative) < negative_size:
            negs = random.choices(group_negative, k=negative_size)
            neg_scores = random.choices(group["negative_scores"], k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
            neg_scores = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negative[:negative_size]
            neg_scores = group["negative_scores"][:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negative)
            negs = [x for x in group_negative]
            neg_scores_all = [x for x in group["negative_scores"]]
            random.Random(_hashed_seed).shuffle(negs)
            random.Random(_hashed_seed).shuffle(neg_scores_all)
            negs = negs * 2
            neg_scores_all = neg_scores_all * 2
            negs = negs[_offset : _offset + negative_size]
            neg_scores = neg_scores_all[_offset : _offset + negative_size]
        for neg_psg, neg_score in zip(negs, neg_scores):
            encoded_passages.append(self.create_one_example(neg_psg))
            negative_scores.append(neg_score)
        return encoded_query, encoded_passages, positive_scores, negative_scores


class EncodeDataset(Dataset):
    input_keys = ["text_id", "text"]

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation="only_first",
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        # pad to the maximum length specified by argument
        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features
