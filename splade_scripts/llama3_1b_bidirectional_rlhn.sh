#!/bin/bash

# Set environment variables before running this script:
export HF_TOKEN=""
export WANDB_API_KEY=""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN='your_token_here'"
    exit 1
fi

# Set wandb variables (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-splade-training}"

export BNB_CUDA_VERSION=124
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TRAINSET=rlhn/rlhn-680K

MODEL_NAME_OR_PATH="<your_merged_pretrained_ckpt>"
TOKENIZER_NAME=meta-llama/Llama-3.2-1B
P_MAX_LEN=192
Q_MAX_LEN=192
Q_MAX_TERMS=512
P_MAX_TERMS=512
Q_FLOPS_FACTOR=0.001
P_FLOPS_FACTOR=0.001
FLOPS_WARMUP=100  # 100 / 4 = 25 steps
USE_HARD_TOPK=False
USE_FLOPS_LOSS=True
LORA_R=32
LORA_ALPHA=64
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULTATION_STEPS=4
LR_SCHEDULER_TYPE=cosine
LR=5e-5
NUM_TRAIN_EPOCHS=1
SAVE_AS=llama3_1b_bi_pretrained_flops_q${Q_FLOPS_FACTOR}_p${P_FLOPS_FACTOR}_bs${TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULTATION_STEPS}_lora${LORA_R}_lr${LR}
RUN_NAME=llama3_1b_bi_pretrained_flops_q${Q_FLOPS_FACTOR}_p${P_FLOPS_FACTOR}_bs${TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULTATION_STEPS}_lora${LORA_R}_lr${LR}

FSDP_CONFIG="accelerate_configs/llama2_mntp_fsdp.yaml"

OUTPUT_DIR=ckpts/model_splade_${SAVE_AS}
CKPT=${OUTPUT_DIR}

accelerate launch --config_file ${FSDP_CONFIG} src/train_splade_causal.py \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--tokenizer_name $TOKENIZER_NAME \
--save_steps 1000 \
--dataset_name $TRAINSET \
--dataset_proc_num 4 \
--lora \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--bf16 \
--negatives_x_device \
--per_device_train_batch_size $TRAIN_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULTATION_STEPS \
--gradient_checkpointing \
--train_n_passages 16 \
--learning_rate $LR \
--lr_scheduler_type $LR_SCHEDULER_TYPE \
--warmup_ratio 0.05 \
--q_max_len $Q_MAX_LEN \
--p_max_len $P_MAX_LEN \
--use_hard_topk ${USE_HARD_TOPK} \
--q_max_terms ${Q_MAX_TERMS} \
--p_max_terms ${P_MAX_TERMS} \
--use_flops_loss ${USE_FLOPS_LOSS} \
--q_flops_loss_factor ${Q_FLOPS_FACTOR} \
--p_flops_loss_factor ${P_FLOPS_FACTOR} \
--flops_warmup ${FLOPS_WARMUP} \
--is_bidirectional True \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--logging_steps 1 \
--run_name $RUN_NAME \
--report_to wandb