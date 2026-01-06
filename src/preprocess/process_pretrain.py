import os
import sys
import json
from tqdm import tqdm

from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict

HF_TOKEN = os.environ.get("HF_TOKEN", "")

new_dataset = []

splits = [
    "wikipedia",
    "gooaq",
    "agnews",
    "ccnews",
    "npr",
    "eli5",
    "cnn",
    "squad",
    "quora",
    "simplewiki",
    "stackexchange_duplicate_questions"
]


for split in splits:
    print(f"Processing split: {split}")

    if split in ["quora", "simplewiki", "stackexchange_duplicate_questions"]:
        query_prefix = "query: "
        document_prefix = "query: "
    else:
        query_prefix = "query: "
        document_prefix = "passage: "

    dataset = load_dataset("nomic-ai/nomic-embed-unsupervised-data", split=split)
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        new_entry = {
            "query_id": f"split_{i}",
            "query": query_prefix + entry["query"],
            "positive_passages": [{"docid": f"split_{i}_pos", "text": document_prefix + entry["document"]}],
            "negative_passages": []
        }

        new_dataset.append(new_entry)

new_dataset = Dataset.from_list(new_dataset)
new_dataset = DatasetDict({"train": new_dataset})

new_dataset.push_to_hub("brutusxu/nomic-embed-pretrain-lite", private=True, token=HF_TOKEN)