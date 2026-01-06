import os
import sys
import argparse
import numpy as np
import json
from collections import defaultdict
import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
from beir import util


def read_ranklist(fname):
    res = defaultdict(dict)
    fin = open(fname, "r")
    for line in fin:
        qid, pid, rank = line.strip().split("\t")
        res[qid][pid] = 1000.0 - int(rank)
    return res


parser = argparse.ArgumentParser()

parser.add_argument("--ranklist", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()

print(f"Evaluating ranklist -> {args.ranklist}")
print(f"Evaluating dataset -> {args.dataset}")

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
out_dir = "./beir_cache"

if not os.path.exists(os.path.join(out_dir, args.dataset)):
    data_path = util.download_and_unzip(url, out_dir)
else:
    data_path = os.path.join(out_dir, args.dataset)

if args.dataset == "msmarco":
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
else:
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

ranklists = read_ranklist(args.ranklist)

evaluator = EvaluateRetrieval()

ndcg, map, recall, precision = evaluator.evaluate(qrels=qrels, results=ranklists, k_values=[10, 100])

print(ndcg)
print(map)
print(recall)
print(precision)
