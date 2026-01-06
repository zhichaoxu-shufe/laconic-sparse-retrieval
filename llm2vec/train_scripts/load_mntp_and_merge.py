import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from peft import PeftModel
from peft import get_peft_model
from peft import LoraConfig
from peft import LoraModel
from llm2vec.models import Qwen2BiForMNTP
from llm2vec.models import Qwen2BiModel
from llm2vec.models import LlamaBiForMNTP
from llm2vec.models import LlamaBiModel

HF_TOKEN=os.environ.get("HF_TOKEN", "")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--lora", type=str, required=True)
parser.add_argument("--save_as", type=str, required=True)
parser.add_argument("--model_cls", type=str, choices=["llama3", "qwen2_5"], required=True)

args = parser.parse_args()
if args.model_cls == "llama3":
    CLS = LlamaBiForMNTP
elif args.model_cls == "qwen2_5":
    CLS = Qwen2BiForMNTP
else:
    raise Exception("model class not supported")

model = CLS.from_pretrained(args.model_name_or_path)

model.model = PeftModel.from_pretrained(
    model.model,
    args.lora
)
model.model = model.model.merge_and_unload()
model.save_pretrained(args.save_as)
