#!/bin/bash

SAVE_AS=$1

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

for DATASET in "${DATASETS[@]}";
do
    echo "---------------------------------"
    echo "Retrieving dataset: ${DATASET}"
    echo "---------------------------------"

CORPUS=./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/corpus_merged/encodings.jsonl
QUERY=./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/query/test.tsv
INDEX_CACHE=./beir_encodings/${DATASET}/encoding_splade_${SAVE_AS}/seismic_index.pkl
RANKLIST=./ranklists/${DATASET}/${SAVE_AS}.tsv

python src/retrieve_from_files_pyseismic.py \
--corpus_files ${CORPUS} \
--query_file ${QUERY} \
--output_path ./ranklists/${DATASET}/${SAVE_AS}.tsv \
--top_k 100 \
--index_cache ${INDEX_CACHE} \
--output_path ${RANKLIST}

echo "---------------------------------"
echo "Evaluating dataset: ${DATASET}"
echo "---------------------------------"

python src/run_eval.py \
--ranklist $RANKLIST \
--dataset $DATASET >> eval_logs/model_based/${SAVE_AS}.txt

done