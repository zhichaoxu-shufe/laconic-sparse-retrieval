#!/bin/bash

TOKENIZER_NAME=$1
SAVE_AS=$2
IS_ECHO=$3
IS_BIDIRECTIONAL=$4
STEP=$5

P_MAX_LEN=256
Q_MAX_LEN=256
P_MAX_TERMS=512
Q_MAX_TERMS=512

if [ -z "$STEP" ]; then
    CKPT=ckpts/model_splade_${SAVE_AS}
else
    CKPT=ckpts/model_splade_${SAVE_AS}/checkpoint-${STEP}
fi

DATASETS=(
    quora
    webis-touche2020
    trec-covid
    climate-fever
    dbpedia-entity
    fever
    hotpotqa
    nq
    msmarco
)


# adjusted order
for DATASET in "${DATASETS[@]}";
do
    mkdir -p beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus
    mkdir -p beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/query

    echo "---------------------------------"
    echo "Encoding dataset: ${DATASET}"
    echo "---------------------------------"

    if [ "$DATASET" = "msmarco" ];
    then
        DATASET_NAME_DOC="Tevatron/msmarco-passage-corpus"
        DATASET_NAME_QRY="Tevatron/msmarco-passage-aug/dev"
    else
        DATASET_NAME_DOC="Tevatron/beir-corpus:${DATASET}"
        DATASET_NAME_QRY="Tevatron/beir:${DATASET}/test"
    fi

    for i in $(seq -f "%02g" 0 7);
    do
        CUDA_VISIBLE_DEVICES=${i} python src/encode_splade_causal.py \
        --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS} \
        --model_name_or_path $CKPT \
        --tokenizer_name $TOKENIZER_NAME \
        --is_echo ${IS_ECHO} \
        --is_bidirectional ${IS_BIDIRECTIONAL} \
        --fp16 \
        --p_max_len $P_MAX_LEN \
        --p_max_terms $P_MAX_TERMS \
        --per_device_eval_batch_size 8 \
        --dataset_name ${DATASET_NAME_DOC} \
        --encode_num_shard 8 \
        --encode_shard_index ${i} \
        --encoded_save_path ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus/split${i}.jsonl &
    done
    wait

    CUDA_VISIBLE_DEVICES=0 python src/encode_splade_causal.py \
    --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS} \
    --model_name_or_path $CKPT \
    --tokenizer_name $TOKENIZER_NAME \
    --is_echo ${IS_ECHO} \
    --is_bidirectional ${IS_BIDIRECTIONAL} \
    --fp16 \
    --q_max_len $Q_MAX_LEN \
    --q_max_terms $Q_MAX_TERMS \
    --encode_is_qry \
    --per_device_eval_batch_size 8 \
    --dataset_name ${DATASET_NAME_QRY} \
    --encoded_save_path ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/query/test.tsv

    mkdir -p ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged

    python src/merge_encodings.py \
    --input_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus/ \
    --output_dir ./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged


done
