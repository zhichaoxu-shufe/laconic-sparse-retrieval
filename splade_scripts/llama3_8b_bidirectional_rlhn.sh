#!/bin/bash
#SBATCH --job-name=<your_job_name>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=<your_partition_name>

set -x -e

export BNB_CUDA_VERSION=124
export NCCL_ASYNC_ERROR_HANDLING=1
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

TRAINSET=rlhn/rlhn-680K

MODEL_NAME_OR_PATH="<your_merged_pretrained_ckpt>"
TOKENIZER_NAME=meta-llama/Llama-3.1-8B
P_MAX_LEN=192
Q_MAX_LEN=192
P_MAX_TERMS=512
Q_MAX_TERMS=512
USE_HARD_TOPK=False
USE_FLOPS_LOSS=True
P_FLOPS_FACTOR=0.001
Q_FLOPS_FACTOR=0.001
FLOPS_WARMUP=100  # 100 / 4 = 25 steps

TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULTATION_STEPS=4
LORA_R=16
LORA_ALPHA=32
LR_SCHEDULER_TYPE=cosine
LR=1e-4
NUM_TRAIN_EPOCHS=1
SAVE_AS=llama3_8b_bi_pretrained_flops_q${Q_FLOPS_FACTOR}_p${P_FLOPS_FACTOR}_bs${TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULTATION_STEPS}_lora${LORA_R}_lr${LR}
RUN_NAME=llama3_8b_bi_pretrained_flops_q${Q_FLOPS_FACTOR}_p${P_FLOPS_FACTOR}_bs${TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULTATION_STEPS}_lora${LORA_R}_lr${LR}


export LAUNCHER="accelerate launch \
    --config_file accelerate_configs/llama2_mntp_fsdp.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
"

export PROGRAM="\
src/train_splade_causal.py \
--output_dir ckpts/model_splade_${SAVE_AS} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--tokenizer_name ${TOKENIZER_NAME} \
--save_steps 1000 \
--dataset_name ${TRAINSET} \
--dataset_proc_num 4 \
--optim adamw_torch \
--lora \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--bf16 \
--negatives_x_device \
--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULTATION_STEPS} \
--gradient_checkpointing \
--train_n_passages 16 \
--learning_rate ${LR} \
--lr_scheduler_type ${LR_SCHEDULER_TYPE} \
--warmup_ratio 0.05 \
--is_bidirectional True \
--q_max_len ${Q_MAX_LEN} \
--p_max_len ${P_MAX_LEN} \
--use_hard_topk ${USE_HARD_TOPK} \
--q_max_terms ${Q_MAX_TERMS} \
--p_max_terms ${P_MAX_TERMS} \
--use_flops_loss ${USE_FLOPS_LOSS} \
--q_flops_loss_factor ${Q_FLOPS_FACTOR} \
--p_flops_loss_factor ${P_FLOPS_FACTOR} \
--flops_warmup ${FLOPS_WARMUP} \
--num_train_epochs ${NUM_TRAIN_EPOCHS} \
--logging_steps 1 \
--run_name ${RUN_NAME} \
--report_to wandb \
"

export CMD="$LAUNCHER $PROGRAM"
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a main_log.txt
echo "END TIME: $(date)"
