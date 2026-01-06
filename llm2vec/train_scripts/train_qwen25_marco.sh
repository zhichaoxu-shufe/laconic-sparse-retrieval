#!/bin/bash

# applying gradient_accumulation_steps leads to problem: 
# [rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 3
# https://github.com/huggingface/transformers/pull/35157

export HF_TOKEN=""

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
SAVE_AS="output/mntp/qwen2.5_1b_instruct_marco_epoch1"

deepspeed --num_gpus=8 \
experiments/run_mntp.py \
--deepspeed train_configs/ds_configs/ds_zero2_config.json \
--model_name_or_path $MODEL_NAME_OR_PATH \
--dataset_name "tevatron/msmarco-passage-corpus" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--do_train \
--max_seq_length 512 \
--mask_token_type "blank" \
--data_collator_type "default" \
--mlm_probability 0.2 \
--output_dir $SAVE_AS \
--eval_strategy "steps" \
--save_steps 1000 \
--num_train_epochs 1 \
--lora_r 16 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_ratio 0.05 \
--bf16 \
--attn_implementation "sdpa" \
--logging_steps 1000 \
--save_only_model 