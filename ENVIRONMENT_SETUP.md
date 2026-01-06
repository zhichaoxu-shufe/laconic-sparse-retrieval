# Environment Variables Setup

This document describes the environment variables required to run the SPLADE training and encoding scripts.

## Required Environment Variables

### HuggingFace Authentication

One of the following environment variables must be set for HuggingFace Hub authentication:

- `HF_TOKEN` (recommended)
- `HUGGING_FACE_HUB_TOKEN` (alternative)

These are used for:
- Downloading models from HuggingFace Hub
- Downloading datasets from HuggingFace Hub
- Pushing models/datasets to HuggingFace Hub (for preprocessing scripts)

**Example:**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Weights & Biases (Optional)

For experiment tracking and logging:

- `WANDB_API_KEY` - Your Weights & Biases API key
- `WANDB_PROJECT` - Project name (default: 'splade-training')
- `WANDB_RUN_NAME` - Run name (default: uses output_dir name)

**Example:**
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="my-splade-experiments"
export WANDB_RUN_NAME="run-001"
```

## Usage in Training Scripts

### Training Script (src/train_splade_causal.py)

The training script will:
1. Check for HuggingFace token and warn if not found
2. Initialize wandb if `WANDB_API_KEY` is set
3. Log training metrics including:
   - Total loss
   - FLOPS losses (query and passage)
   - FLOPS lambda values
   - Per-dimension losses (for Matryoshka training)

### Encoding Script (src/encode_splade_causal.py)

The encoding script requires HuggingFace token for loading models and datasets.

### Preprocessing Scripts

Scripts in `src/preprocess/` require HuggingFace token, especially:
- `process_msmarco_hard.py` - Requires token for pushing to Hub
- `measure_corpus.py` - Requires token for loading datasets

## Example Bash Script Setup

```bash
#!/bin/bash

# HuggingFace Authentication
export HF_TOKEN="hf_your_token_here"

# Weights & Biases (Optional)
export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="splade-experiments"
export WANDB_RUN_NAME="baseline-run"

# Run training
python src/train_splade_causal.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --output_dir "outputs/run1" \
    # ... other arguments
```

## Security Notes

1. **Never commit tokens to version control**
2. Keep your tokens in secure locations (e.g., `.env` files that are gitignored)
3. Use environment variables or secure secret management systems
4. Rotate tokens periodically for security

## Wandb Logging Features

When `WANDB_API_KEY` is set, the training script logs:

### Standard Training Mode
- `train/total_loss` - Combined loss
- `train/main_loss` - Main contrastive loss
- `train/q_flops_loss` - Query FLOPS regularization
- `train/p_flops_loss` - Passage FLOPS regularization
- `train/q_flops_lambda` - Query FLOPS scheduler value
- `train/p_flops_lambda` - Passage FLOPS scheduler value

### Matryoshka Training Mode
- `train/total_loss` - Weighted sum across all dimensions
- `train/loss_q{dim}_p{dim}` - Per-dimension losses
- `train/q_flops_loss` - Optional FLOPS loss on full representations
- `train/p_flops_loss` - Optional FLOPS loss on full representations
- Scheduler lambda values as above

All metrics are logged at each training step for detailed tracking.
