#!/bin/bash

SAVE_AS=$1

DATASETS=(
    "android"
    "english"
    "gaming"
    "gis"
    "mathematica"
    "physics"
    "programmers"
    "stats"
    "tex"
    "unix"
    "webmasters"
    "wordpress"
)

for DATASET in "${DATASETS[@]}";
do

echo "---------------------------------"
echo "Retrieving dataset: cqadupstack-${DATASET}"
echo "---------------------------------"

CORPUS=./beir_encodings/cqadupstack-${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged/encodings.jsonl
QUERY=./beir_encodings/cqadupstack-${DATASET}/encoding_splade_${SAVE_AS}/query/test.tsv
INDEX_CACHE=./beir_encodings/cqadupstack-${DATASET}/encoding_splade_${SAVE_AS}/seismic_index.pkl
RANKLIST=./ranklists/cqadupstack-${DATASET}/${SAVE_AS}.tsv

python src/retrieve_from_files_pyseismic.py \
--corpus_files ${CORPUS} \
--query_file ${QUERY} \
--output_path ./ranklists/cqadupstack-${DATASET}/${SAVE_AS}.tsv \
--top_k 100 \
--index_cache ${INDEX_CACHE} \
--output_path ${RANKLIST}

echo "---------------------------------"
echo "Evaluating dataset: cqadupstack-${DATASET}"
echo "---------------------------------"

python src/run_eval.py \
--ranklist $RANKLIST \
--dataset cqadupstack/${DATASET} >> eval_logs/model_based/${SAVE_AS}_cqa.txt

done