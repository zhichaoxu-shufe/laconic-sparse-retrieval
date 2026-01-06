import os
import sys
import time
import argparse

from transformers import AutoModelForCausalLM
from transformers import AutoModel
from llm2vec.models import LlamaBiForMNTP
from llm2vec.models import Qwen2BiForMNTP
from peft import PeftModel


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--adapter", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--is_causal_lm", action="store_true")
parser.add_argument("--is_bidirectional", action="store_true")

args = parser.parse_args()

MODEL_CLS = AutoModelForCausalLM if args.is_causal_lm else AutoModel
if args.is_bidirectional:
    if "llama" in args.base_model.lower():
        MODEL_CLS = LlamaBiForMNTP
    elif "qwen" in args.base_model.lower():
        MODEL_CLS = Qwen2BiForMNTP

base_model = MODEL_CLS.from_pretrained(args.base_model)
peft_model_id = args.adapter

model = PeftModel.from_pretrained(base_model, peft_model_id)

model = model.merge_and_unload()

if args.output_dir == "":
    args.output_dir = f"{args.adapter}_merged"

model.save_pretrained(args.output_dir)