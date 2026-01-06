#!/bin/bash
#SBATCH --job-name=<your_job_name>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=<your_partition>

set -x -e

export BNB_CUDA_VERSION=124
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "START TIME: $(date)"

# Set environment variables before running this script:
export HF_TOKEN=""
export WANDB_API_KEY=""

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000


# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN='your_token_here'"
    exit 1
fi

# Set wandb variables (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-splade-training}"

TRAINSET=brutusxu/nomic-embed-pretrain-lite

MODEL_NAME_OR_PATH=meta-llama/Llama-3.2-1B
TOKENIZER_NAME=meta-llama/Llama-3.2-1B
P_MAX_LEN=192
Q_MAX_LEN=64
Q_MAX_TERMS=512
P_MAX_TERMS=512
Q_FLOPS_FACTOR=0.0005
P_FLOPS_FACTOR=0.0005
FLOPS_WARMUP=4000  # 4000 / 4 = 1000 steps
LORA_R=32
LORA_ALPHA=64
TRAIN_BATCH_SIZE=64  # 4x8x64=2048 effective batch size, 1130 steps per epoch
GRADIENT_ACCUMULTATION_STEPS=4
LR_SCHEDULER_TYPE=cosine
LR=1e-4
NUM_TRAIN_EPOCHS=3
SAVE_AS=llama3_1b_bi_pretrain_epoch3
RUN_NAME=llama3_1b_bi_pretrain_epoch3

FSDP_CONFIG="accelerate_configs/llama2_mntp_fsdp.yaml"

OUTPUT_DIR=ckpts/model_msmarco_splade_${SAVE_AS}
CKPT=${OUTPUT_DIR}

export LAUNCHER="accelerate launch \
    --config_file ${FSDP_CONFIG} \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
"

export PROGRAM="\
src/train_splade_causal.py \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--tokenizer_name $TOKENIZER_NAME \
--save_steps 500 \
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
--train_n_passages 1 \
--learning_rate $LR \
--lr_scheduler_type $LR_SCHEDULER_TYPE \
--warmup_ratio 0.05 \
--q_max_len $Q_MAX_LEN \
--p_max_len $P_MAX_LEN \
--use_hard_topk False \
--q_max_terms $Q_MAX_TERMS \
--p_max_terms $P_MAX_TERMS \
--use_flops_loss True \
--q_flops_loss_factor ${Q_FLOPS_FACTOR} \
--p_flops_loss_factor ${P_FLOPS_FACTOR} \
--is_bidirectional True \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--logging_steps 1 \
--run_name $RUN_NAME \
--report_to wandb \
"

export CMD="$LAUNCHER $PROGRAM"
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a main_log.txt
echo "END TIME: $(date)"