# PySeismic Retrieval Mode Documentation

## Overview

The `encode_splade_causal_pyseismic.py` script now supports two modes:

1. **Original Mode**: Single-dataset encoding (backward compatible)
2. **Retrieval Mode**: Dual-dataset encoding with in-memory retrieval using `semantic_search_seismic`

## Retrieval Mode

### How It Works

When both `QUERY_DATASET_NAME` and `DOCUMENT_DATASET_NAME` environment variables are set, the script enters **Retrieval Mode**:

1. **Encode Corpus**: Loads and encodes documents from HuggingFace dataset
2. **Encode Queries**: Loads and encodes queries from HuggingFace dataset  
3. **Retrieve**: Performs in-memory semantic search using PySeismic
4. **Output**: Writes results to TSV file in format: `qid\tpid\trank`

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QUERY_DATASET_NAME` | HuggingFace dataset name for queries | None | Yes (for retrieval mode) |
| `DOCUMENT_DATASET_NAME` | HuggingFace dataset name for documents | None | Yes (for retrieval mode) |
| `RETRIEVAL_TOP_K` | Number of documents to retrieve per query | 100 | No |

### Usage Example

```bash
# Set environment variables
export QUERY_DATASET_NAME="Tevatron/beir:nfcorpus/test"
export DOCUMENT_DATASET_NAME="Tevatron/beir-corpus:nfcorpus"
export RETRIEVAL_TOP_K=100

# Run retrieval
CUDA_VISIBLE_DEVICES=0 python src/encode_splade_causal_pyseismic.py \
    --output_dir ./results/nfcorpus \
    --model_name_or_path /path/to/checkpoint \
    --tokenizer_name /path/to/tokenizer \
    --fp16 \
    --q_max_len 32 \
    --p_max_len 256 \
    --per_device_eval_batch_size 32 \
    --dataset_name dummy \
    --encoded_save_path ./results/nfcorpus/rank.tsv

# Clean up
unset QUERY_DATASET_NAME
unset DOCUMENT_DATASET_NAME
unset RETRIEVAL_TOP_K
```

### Output Format

The rank file is a TSV with three columns:
```
qid    pid    rank
101    doc_1  1
101    doc_2  2
101    doc_3  3
102    doc_5  1
...
```

Where:
- `qid`: Query ID
- `pid`: Document/Passage ID
- `rank`: Rank position (starts at 1)

## Key Differences from Original Mode

### Sparse Representation

**Retrieval Mode:**
- Float values (no quantization)
- Format: `{token: float_weight, ...}`
- Compatible with `semantic_search_seismic`

**Original Mode:**
- Integer quantization: `np.rint(data * 100).astype(int)`
- Format: `{token: int_weight, ...}`
- Compatible with Lucene/Anserini

### Top-K Sparsification

Both modes support:
- Queries: 128 dimensions (default)
- Documents: 1024 dimensions (default)
- Custom: Set via `matryoshka_dim` parameter

### Dataset Loading

**Retrieval Mode:**
- Loads from HuggingFace datasets via environment variables
- Encodes both queries and documents in single run
- No sharding for queries (full dataset)
- Optional sharding for documents

**Original Mode:**
- Loads from `dataset_name` parameter
- Encodes either queries OR documents
- Supports sharding for both

## Performance Notes

### Memory Requirements

Retrieval mode stores all embeddings in memory:
- Queries: ~128 non-zero terms × 4 bytes × num_queries
- Documents: ~1024 non-zero terms × 4 bytes × num_docs

Example for BEIR NFCorpus:
- 323 queries × 128 terms = ~165 KB
- 3,633 documents × 1024 terms = ~15 MB
- **Total: ~15 MB** (very manageable)

For larger corpora (e.g., MS MARCO):
- 6,980 queries × 128 terms = ~3.5 MB
- 8.8M documents × 1024 terms = ~36 GB
- Consider sharding for very large corpora

### Speed

PySeismic is highly optimized:
- NFCorpus (3.6K docs): ~0.1 seconds
- FiQA (57K docs): ~1 second
- Larger datasets: Linear scaling with corpus size

## Backward Compatibility

The original single-dataset encoding mode remains fully functional:

```bash
# Original mode (no environment variables set)
python src/encode_splade_causal_pyseismic.py \
    --model_name_or_path /path/to/checkpoint \
    --dataset_name Tevatron/beir-corpus:nfcorpus \
    --encode_is_qry \
    --encoded_save_path output.tsv
```

## Evaluation

After generating the rank file, evaluate using your preferred metrics:

```bash
# Example with pytrec_eval or similar tools
python -m pytrec_eval qrels.tsv rank.tsv
```

## Troubleshooting

### Empty Embeddings

If you see warnings about empty embeddings:
```
WARNING:__main__:Empty corpus embedding for ID: 12345
```

The script automatically adds a fallback token (pad_token) with weight 1.0 to prevent issues.

### Memory Errors

For very large corpora that don't fit in memory, consider:
1. Using sharding: `--encode_num_shard 8 --encode_shard_index 0`
2. Processing in batches and combining results
3. Using the original mode with external indexing (Lucene)

### Dataset Not Found

Ensure your dataset name follows HuggingFace convention:
- Corpus: `Tevatron/beir-corpus:dataset_name`
- Queries: `Tevatron/beir:dataset_name/split`

## Examples

### BEIR Datasets

```bash
# NFCorpus
export QUERY_DATASET_NAME="Tevatron/beir:nfcorpus/test"
export DOCUMENT_DATASET_NAME="Tevatron/beir-corpus:nfcorpus"

# FiQA
export QUERY_DATASET_NAME="Tevatron/beir:fiqa/test"
export DOCUMENT_DATASET_NAME="Tevatron/beir-corpus:fiqa"

# SciFact
export QUERY_DATASET_NAME="Tevatron/beir:scifact/test"
export DOCUMENT_DATASET_NAME="Tevatron/beir-corpus:scifact"
```

### Custom Top-K

```bash
export RETRIEVAL_TOP_K=1000  # Retrieve top-1000 instead of default 100
```

### With Matryoshka Dimensions

```bash
# Add to command line args
--matryoshka_dim 256  # Use 256 dimensions instead of defaults
