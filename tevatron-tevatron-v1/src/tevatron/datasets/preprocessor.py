import os
import sys


class TrainPreProcessor:
    """
    As discussed in https://arxiv.org/pdf/2304.12904, training on titled dataset leads to performance
    degradation on DL19 and DL20 dataset.
    """

    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=" ", q_prefix="query: ", p_prefix="passage: ", lowercase=False, add_eos_token=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.q_prefix = q_prefix
        self.p_prefix = p_prefix
        self.lowercase = lowercase
        self.add_eos_token = add_eos_token
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    def _encode_with_eos(self, text, max_length):
        """Encode text and optionally append EOS token."""
        if self.add_eos_token and self.eos_token_id is not None:
            # Reserve space for EOS token
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 1,
                truncation=True,
            )
            # Append EOS token
            encoded.append(self.eos_token_id)
        else:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        return encoded

    def __call__(self, example):
        query_text = example["query"]
        if self.lowercase:
            query_text = query_text.lower()
        
        query = self._encode_with_eos(
            self.q_prefix + query_text,
            self.query_max_length
        )
        
        positives = []
        for pos in example["positive_passages"]:
            text = (
                pos["title"] + self.separator + pos["text"]
                if ("title" in pos and pos["title"] != None)
                else pos["text"]
            )
            if self.lowercase:
                text = text.lower()
            positives.append(
                self._encode_with_eos(
                    self.p_prefix + text,
                    self.text_max_length
                )
            )
        
        negatives = []
        for neg in example["negative_passages"]:
            text = (
                neg["title"] + self.separator + neg["text"]
                if ("title" in neg and neg["title"] != None)
                else neg["text"]
            )
            if self.lowercase:
                text = text.lower()
            negatives.append(
                self._encode_with_eos(
                    self.p_prefix + text,
                    self.text_max_length
                )
            )
        
        return {"query": query, "positives": positives, "negatives": negatives}


class DistillationTrainPreProcessor:
    """Preprocessor for distillation training with query, positive and negative passages."""

    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=" ", q_prefix="query: ", p_prefix="passage: ", lowercase=False, add_eos_token=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.q_prefix = q_prefix
        self.p_prefix = p_prefix
        self.lowercase = lowercase
        self.add_eos_token = add_eos_token
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    def _encode_with_eos(self, text, max_length):
        """Encode text and optionally append EOS token."""
        if self.add_eos_token and self.eos_token_id is not None:
            # Reserve space for EOS token
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 1,
                truncation=True,
            )
            # Append EOS token
            encoded.append(self.eos_token_id)
        else:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        return encoded

    def __call__(self, example):
        query_text = example["query"]
        if self.lowercase:
            query_text = query_text.lower()
        
        query = self._encode_with_eos(
            self.q_prefix + query_text,
            self.query_max_length
        )
        
        positives = []
        positive_scores = []
        for pos in example["positive_passages"]:
            text = (
                pos["title"] + self.separator + pos["text"]
                if ("title" in pos and pos["title"] != None)
                else pos["text"]
            )
            if self.lowercase:
                text = text.lower()
            positives.append(
                self._encode_with_eos(
                    self.p_prefix + text,
                    self.text_max_length
                )
            )
            positive_scores.append(pos["score"])
        
        negatives = []
        negative_scores = []
        for neg in example["negative_passages"]:
            text = (
                neg["title"] + self.separator + neg["text"]
                if ("title" in neg and neg["title"] != None)
                else neg["text"]
            )
            if self.lowercase:
                text = text.lower()
            negatives.append(
                self._encode_with_eos(
                    self.p_prefix + text,
                    self.text_max_length
                )
            )
            negative_scores.append(neg["score"])
        
        return {
            "query": query, 
            "positives": positives, 
            "negatives": negatives, 
            "positive_scores": positive_scores, 
            "negative_scores": negative_scores
        }


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, q_prefix="query: ", lowercase=False, add_eos_token=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.q_prefix = "query: "
        self.lowercase = lowercase
        self.add_eos_token = add_eos_token
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    def _encode_with_eos(self, text, max_length):
        """Encode text and optionally append EOS token."""
        if self.add_eos_token and self.eos_token_id is not None:
            # Reserve space for EOS token
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 1,
                truncation=True,
            )
            # Append EOS token
            encoded.append(self.eos_token_id)
        else:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        return encoded

    def __call__(self, example):
        query_id = example["query_id"]
        query_text = example["query"]
        if self.lowercase:
            query_text = query_text.lower()
        
        query = self._encode_with_eos(
            self.q_prefix + query_text,
            self.query_max_length
        )
        return {"text_id": query_id, "text": query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=" ", p_prefix="passage: ", lowercase=False, add_eos_token=False):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator
        self.p_prefix = p_prefix
        self.lowercase = lowercase
        self.add_eos_token = add_eos_token
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    def _encode_with_eos(self, text, max_length):
        """Encode text and optionally append EOS token."""
        if self.add_eos_token and self.eos_token_id is not None:
            # Reserve space for EOS token
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 1,
                truncation=True,
            )
            # Append EOS token
            encoded.append(self.eos_token_id)
        else:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        return encoded

    def __call__(self, example):
        docid = example["docid"]
        text = (
            example["title"] + self.separator + example["text"]
            if ("title" in example and example["title"] != None)
            else example["text"]
        )
        
        if self.lowercase:
            text = text.lower()

        text = self._encode_with_eos(
            self.p_prefix + text,
            self.text_max_length
        )
        return {"text_id": docid, "text": text}


class ContrastivePreProcessor:
    """
    Flexible preprocessor for two-stage training:
    1. Contrastive Pretraining: Handles query-document pairs (no explicit negatives)
    2. Finetuning: Handles query-pos-neg triplets
    
    When negative_passages is empty or missing, returns empty negatives list for in-batch negative training.
    """

    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=" ", q_prefix="", p_prefix="", lowercase=False, add_eos_token=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.q_prefix = q_prefix
        self.p_prefix = p_prefix
        self.lowercase = lowercase
        self.add_eos_token = add_eos_token
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

    def _encode_with_eos(self, text, max_length):
        """Encode text and optionally append EOS token."""
        if self.add_eos_token and self.eos_token_id is not None:
            # Reserve space for EOS token
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length - 1,
                truncation=True,
            )
            # Append EOS token
            encoded.append(self.eos_token_id)
        else:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        return encoded

    def __call__(self, example):
        # Process query
        query_text = example["query"]
        if self.lowercase:
            query_text = query_text.lower()
        
        query = self._encode_with_eos(
            self.q_prefix + query_text,
            self.query_max_length
        )
        
        # Process positive passages
        positives = []
        for pos in example["positive_passages"]:
            text = (
                pos["title"] + self.separator + pos["text"]
                if ("title" in pos and pos["title"] != None)
                else pos["text"]
            )
            if self.lowercase:
                text = text.lower()
            positives.append(
                self._encode_with_eos(
                    self.p_prefix + text,
                    self.text_max_length
                )
            )
        
        # Process negative passages (if they exist)
        negatives = []
        if "negative_passages" in example and example["negative_passages"]:
            for neg in example["negative_passages"]:
                text = (
                    neg["title"] + self.separator + neg["text"]
                    if ("title" in neg and neg["title"] != None)
                    else neg["text"]
                )
                if self.lowercase:
                    text = text.lower()
                negatives.append(
                    self._encode_with_eos(
                        self.p_prefix + text,
                        self.text_max_length
                    )
                )
        
        return {"query": query, "positives": positives, "negatives": negatives}
