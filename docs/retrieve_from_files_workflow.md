# Retrieve from Pre-encoded Files Workflow

## Overview

This document describes the workflow for encoding queries and documents separately, then performing retrieval using PySeismic on pre-encoded files.

## Workflow

### Step 1: Encode Corpus (Separate Process)

Use the original encoding script to encode documents in parallel (can use multiple GPUs):

```bash
# Encode corpus with 8 shards (can run in parallel)
for i in $(seq -f "%02g" 0 7); do
    CUDA_VISIBLE_DEVICES=${i} python src/encode_splade_causal.py \
        --output_dir ./beir_encodings/arguana/encoding_splade_model \
        --model_name_or_path /path/to/checkpoint \
        --tokenizer_name meta-llama/Llama-3.2-1B \
        --fp16 \
        --p_max_len 512 \
        --per_device_eval_batch_size 32 \
        --dataset_name Tevatron/beir-corpus:arguana \
        --encode_num_shard 8 \
        --encode_shard_index ${i} \
        --encoded_save_path ./beir_encodings/arguana/encoding_splade_model/corpus/split${i}.jsonl &
done
wait
```

**Output:** `./beir_encodings/arguana/encoding_splade_model/corpus/split*.jsonl`

### Step 2: Encode Queries (Single Process)

```bash
CUDA_VISIBLE_DEVICES=0 python src/encode_splade_causal.py \
    --output_dir ./beir_encodings/arguana/encoding_splade_model \
    --model_name_or_path /path/to/checkpoint \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --fp16 \
    --q_max_len 256 \
    --encode_is_qry \
    --per_device_eval_batch_size 32 \
    --dataset_name Tevatron/beir:arguana/test \
    --encoded_save_path ./beir_encodings/arguana/encoding_splade_model/query/test.tsv
```

**Output:** `./beir_encodings/arguana/encoding_splade_model/query/test.tsv`

### Step 3: Retrieve with PySeismic

Now use the new `retrieve_from_files_pyseismic.py` script:

```bash
python src/retrieve_from_files_pyseismic.py \
    --corpus_files ./beir_encodings/arguana/encoding_splade_model/corpus/*.jsonl \
    --query_file ./beir_encodings/arguana/encoding_splade_model/query/test.tsv \
    --output_path ./ranklists/arguana/model.tsv \
    --top_k 100 \
    --index_cache ./beir_encodings/arguana/encoding_splade_model/seismic_index.pkl
```

**Or use the convenience script:**

```bash
bash splade_scripts/eval_scripts/retrieve_from_files.sh arguana model_name
```

## Float Weights vs Integer Quantization

### New Feature: Float Precision (Default)

By default, encoding now uses **float weights** for better precision:

```bash
# Float weights (default, use_float_weights=True)
python src/encode_splade_causal.py \
    --model_name_or_path ckpts/my_model \
    --dataset_name Tevatron/beir:arguana/test \
    --encode_is_qry \
    --encoded_save_path query/test.tsv
```

**Output format:**
- Corpus: `{"id": "doc1", "content": "", "vector": {"token1": 0.156, "token2": 0.089}}`
- Query: `query_001\t{"token1": 0.142, "token2": 0.201}`

### Legacy: Integer Quantization

For Lucene/Anserini compatibility, use `--use_float_weights False`:

```bash
# Integer quantization (use_float_weights=False)
python src/encode_splade_causal.py \
    --model_name_or_path ckpts/my_model \
    --dataset_name Tevatron/beir:arguana/test \
    --encode_is_qry \
    --use_float_weights False \
    --encoded_save_path query/test.tsv
```

**Output format:**
- Corpus: `{"id": "doc1", "content": "", "vector": {"token1": 15, "token2": 8}}`
- Query: `query_001\ttoken1 token1 token1 token2 token2`

### Why Float Weights?

**Precision Loss Example with Integer Quantization:**
- Original: `0.0537` → Quantized: `5` → Reconstructed: `0.05`
- Original: `0.0542` → Quantized: `5` → Reconstructed: `0.05`
- Original: `0.0549` → Quantized: `5` → Reconstructed: `0.05`

Three distinct values become identical! **Float weights preserve precision** for better retrieval quality.

## File Formats

### Corpus Files (.jsonl)

Each line is a JSON object:
```json
{"id": "doc_123", "content": "", "vector": {"token1": 42, "token2": 15, "token3": 8}}
```

- `id`: Document ID (string)
- `content`: Empty string (not used)
- `vector`: Sparse representation (dict mapping token to integer weight)

### Query Files (.tsv)

Two formats are supported:

**Format 1: JSON Dict (Float Weights) - Default**
```
query_001	{"machine": 0.156, "learning": 0.142, "deep": 0.089}
query_002	{"neural": 0.201, "network": 0.183, "architecture": 0.098}
```

**Format 2: Repeated Tokens (Integer Weights) - Legacy**
```
query_001	what is machine learning machine learning deep
query_002	neural network architecture architecture
```

- Column 1: Query ID
- Column 2: Either JSON dict (float) or space-separated repeated tokens (int)

The retrieval script automatically detects which format is used.

### Output Rank File (.tsv)

Standard TREC format:
```
query_001	doc_123	1
query_001	doc_456	2
query_001	doc_789	3
```

- Column 1: Query ID
- Column 2: Document ID
- Column 3: Rank (starts at 1)

## Features

### 1. Multi-Shard Corpus Support

Load corpus from multiple sharded files:

```bash
python src/retrieve_from_files_pyseismic.py \
    --corpus_files corpus/split00.jsonl corpus/split01.jsonl corpus/split02.jsonl \
    --query_file queries/test.tsv \
    --output_path results/rank.tsv
```

Or use shell glob:
```bash
--corpus_files corpus/*.jsonl
```

### 2. Index Caching

Save/load PySeismic index for faster repeated retrievals:

**First run (builds and saves index):**
```bash
python src/retrieve_from_files_pyseismic.py \
    --corpus_files corpus/*.jsonl \
    --query_file queries/test.tsv \
    --index_cache seismic_index.pkl \
    --output_path results/rank_test.tsv
```

**Subsequent runs (loads cached index - much faster!):**
```bash
python src/retrieve_from_files_pyseismic.py \
    --corpus_files corpus/*.jsonl \
    --query_file queries/dev.tsv \
    --index_cache seismic_index.pkl \
    --output_path results/rank_dev.tsv
```

### 3. Force Rebuild Index

Force rebuilding the index even if cache exists:

```bash
python src/retrieve_from_files_pyseismic.py \
    --corpus_files corpus/*.jsonl \
    --query_file queries/test.tsv \
    --index_cache seismic_index.pkl \
    --force_rebuild_index \
    --output_path results/rank.tsv
```

## Performance Notes

### Memory Usage

**Without index caching:**
- Loads all corpus embeddings into memory
- Memory = ~20-25 KB per document
- Example: 8.8M docs = ~176 GB

**With index caching:**
- Index file size: ~100-150 GB for 8.8M docs
- Loading is much faster than rebuilding
- Subsequent retrievals only load the index

### Speed

**First run (building index):**
- Depends on corpus size
- Example: 8.8M docs takes ~10-20 minutes

**Subsequent runs (loading cached index):**
- Much faster: ~30 seconds to load index
- Retrieval: ~1-10 seconds depending on query count

### Recommendations for Large Corpora

For corpora with millions of documents:

1. **Use sharded encoding** (8 shards recommended)
2. **Build index once** and cache it
3. **Reuse cached index** for multiple query sets
4. Consider **distributed retrieval** if single-machine memory is insufficient

## Complete Example

### Scenario: Evaluate on BEIR ArgUANA

```bash
# 1. Encode corpus (8 shards in parallel)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=${i} python src/encode_splade_causal.py \
        --model_name_or_path ckpts/my_model \
        --dataset_name Tevatron/beir-corpus:arguana \
        --encode_num_shard 8 \
        --encode_shard_index ${i} \
        --encoded_save_path beir_encodings/arguana/corpus/split${i}.jsonl \
        --fp16 --p_max_len 512 --per_device_eval_batch_size 32 &
done
wait

# 2. Encode queries
CUDA_VISIBLE_DEVICES=0 python src/encode_splade_causal.py \
    --model_name_or_path ckpts/my_model \
    --dataset_name Tevatron/beir:arguana/test \
    --encode_is_qry \
    --encoded_save_path beir_encodings/arguana/query/test.tsv \
    --fp16 --q_max_len 256 --per_device_eval_batch_size 32

# 3. Retrieve
python src/retrieve_from_files_pyseismic.py \
    --corpus_files beir_encodings/arguana/corpus/*.jsonl \
    --query_file beir_encodings/arguana/query/test.tsv \
    --output_path ranklists/arguana/my_model.tsv \
    --top_k 100 \
    --index_cache beir_encodings/arguana/seismic_index.pkl

# 4. Evaluate
python -m pytrec_eval qrels/arguana.tsv ranklists/arguana/my_model.tsv
```

## Advantages Over Original Workflow

### Old Workflow (Lucene-based)
1. Encode queries and docs separately
2. Build Lucene index from encoded docs
3. Search Lucene index with encoded queries
4. Requires Java, Anserini/Lucene setup

### New Workflow (PySeismic-based)
1. Encode queries and docs separately ✓ (same)
2. Load embeddings directly into memory
3. Build PySeismic index (pure Python)
4. Search with PySeismic (pure Python)
5. **No external dependencies** (Java, Lucene)
6. **Faster iteration** (no index building step)
7. **Index caching** for repeated retrievals

## Troubleshooting

### Out of Memory

If you run out of memory loading the full corpus:

1. Use sharding and process shards separately
2. Implement shard-level retrieval and merge results
3. Consider using a machine with more RAM
4. Use distributed retrieval across multiple machines

### Slow Index Building

First-time index building can be slow for large corpora. Solutions:

1. Use `--index_cache` to save the index
2. Reuse the cached index for subsequent retrievals
3. Build index once, retrieve many times

### File Not Found Errors

Ensure your file paths are correct:

```bash
# Check if files exist
ls -lh beir_encodings/arguana/corpus/*.jsonl
ls -lh beir_encodings/arguana/query/test.tsv
```

## Summary

The new `retrieve_from_files_pyseismic.py` script provides:

- ✅ Clean separation of encoding and retrieval
- ✅ Multi-shard corpus support
- ✅ Index caching for fast repeated retrievals
- ✅ Pure Python implementation (no Java/Lucene)
- ✅ Memory-efficient loading
- ✅ Easy to use and integrate into pipelines
