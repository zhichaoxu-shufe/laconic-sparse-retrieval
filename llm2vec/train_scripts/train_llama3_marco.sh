#!/bin/bash

# applying gradient_accumulation_steps leads to problem: 
# [rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 3
# https://github.com/huggingface/transformers/pull/35157

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# SAVE_AS="output/mntp/qwen2.5_0.5b_marco"

MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B-Instruct"
SAVE_AS="output/mntp/llama3_1b_instruct_marco_epoch1"

# deepspeed --num_gpus=8 \
# experiments/run_mntp.py \
# --deepspeed train_configs/ds_configs/ds_zero2_config.json \
# --model_name_or_path $MODEL_NAME_OR_PATH \
# --dataset_name "tevatron/msmarco-passage-corpus" \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --gradient_accumulation_steps 1 \
# --gradient_checkpointing \
# --do_train \
# --max_seq_length 512 \
# --mask_token_type "blank" \
# --data_collator_type "default" \
# --mlm_probability 0.2 \
# --overwrite_output_dir \
# --output_dir $SAVE_AS \
# --eval_strategy "steps" \
# --save_steps 100 \
# --max_steps 1000 \
# --lora_r 16 \
# --learning_rate 5e-5 \
# --lr_scheduler_type "cosine" \
# --warmup_ratio 0.05 \
# --bf16 \
# --attn_implementation "flash_attention_2" \
# --logging_steps 100 \
# --save_only_model

# for large scale training
deepspeed --num_gpus=8 \
experiments/run_mntp.py \
--deepspeed train_configs/ds_configs/ds_zero2_config.json \
--model_name_or_path $MODEL_NAME_OR_PATH \
--dataset_name "tevatron/msmarco-passage-corpus" \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
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