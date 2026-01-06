#!/bin/bash

TOKENIZER=$1
SAVE_AS=$2
IS_ECHO=$3
IS_BIDIRECTIONAL=$4
STEP=$5

export BNB_CUDA_VERSION=124
OUTPUT_DIR="./ranklists/${DATASET}"

P_MAX_LEN=256
Q_MAX_LEN=256
P_MAX_TERMS=512
Q_MAX_TERMS=512

ADD_EOS_TOKEN=False
POOLING_STRATEGY="max"

if [ -z "$STEP" ]; then
    CKPT=ckpts/model_splade_${SAVE_AS}
else
    CKPT=ckpts/model_splade_${SAVE_AS}/checkpoint-${STEP}
fi

echo "loading checkpoint from ${CKPT}"


# Environment variables for retrieval mode
export RETRIEVAL_TOP_K=100

DATASETS=(
    "cqadupstack-android"
    "cqadupstack-english"
    "cqadupstack-gaming"
    "cqadupstack-gis"
    "cqadupstack-mathematica"
    "cqadupstack-physics"
    "cqadupstack-programmers"
    "cqadupstack-stats"
    "cqadupstack-tex"
    "cqadupstack-unix"
    "cqadupstack-webmasters"
    "cqadupstack-wordpress"
)

for DATASET in "${DATASETS[@]}";
do
    mkdir -p beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus
    mkdir -p beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/query

    for i in $(seq -f "%02g" 0 7);
    do
        CUDA_VISIBLE_DEVICES=${i} python src/encode_splade_causal.py \
        --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS} \
        --model_name_or_path $CKPT \
        --tokenizer_name $TOKENIZER \
        --is_echo $IS_ECHO \
        --is_bidirectional $IS_BIDIRECTIONAL \
        --fp16 \
        --p_max_len $P_MAX_LEN \
        --p_max_terms $P_MAX_TERMS \
        --per_device_eval_batch_size 8 \
        --dataset_name Tevatron/beir-corpus:${DATASET} \
        --encode_num_shard 8 \
        --encode_shard_index ${i} \
        --encoded_save_path ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus/split${i}.jsonl &
    done
    wait

    CUDA_VISIBLE_DEVICES=0 python src/encode_splade_causal.py \
    --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS} \
    --model_name_or_path $CKPT \
    --tokenizer_name $TOKENIZER \
    --is_echo $IS_ECHO \
    --is_bidirectional $IS_BIDIRECTIONAL \
    --fp16 \
    --q_max_len $Q_MAX_LEN \
    --q_max_terms $Q_MAX_TERMS \
    --encode_is_qry \
    --per_device_eval_batch_size 8 \
    --dataset_name Tevatron/beir:${DATASET}/test \
    --encoded_save_path ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/query/test.tsv

    mkdir -p ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged

    python src/merge_encodings.py \
    --input_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus/ \
    --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged

done