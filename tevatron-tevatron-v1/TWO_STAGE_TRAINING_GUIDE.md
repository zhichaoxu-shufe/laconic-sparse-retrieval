# Two-Stage Training Guide: Contrastive Pretraining + Finetuning

This guide explains how to use the new `ContrastivePreProcessor` for two-stage training in Tevatron.

## Overview

The `ContrastivePreProcessor` supports both training stages:

1. **Stage 1 - Contrastive Pretraining**: Learn representations using query-document pairs with in-batch negatives
2. **Stage 2 - Finetuning**: Refine model using query-pos-neg triplets with hard negatives

## Data Format Requirements

### Stage 1: Contrastive Pretraining Data

Your dataset should have this format (no explicit negatives required):

```json
{
  "query": "what is python programming",
  "positive_passages": [
    {
      "title": "Python (programming language)",
      "text": "Python is a high-level, interpreted programming language..."
    }
  ]
}
```

Or without titles:

```json
{
  "query": "what is python programming",
  "positive_passages": [
    {
      "text": "Python is a high-level, interpreted programming language..."
    }
  ]
}
```

### Stage 2: Finetuning Data

Your dataset should include hard negatives:

```json
{
  "query": "what is python programming",
  "positive_passages": [
    {
      "title": "Python (programming language)",
      "text": "Python is a high-level, interpreted programming language..."
    }
  ],
  "negative_passages": [
    {
      "title": "Snake",
      "text": "Pythons are non-venomous snakes found in Africa, Asia..."
    },
    {
      "title": "Monty Python",
      "text": "Monty Python is a British comedy group..."
    }
    // ... more hard negatives
  ]
}
```

## Usage

### Method 1: Using PROCESSOR_INFO Dictionary (Recommended)

Add your datasets to the `PROCESSOR_INFO` in `dataset.py`:

```python
PROCESSOR_INFO = {
    # ... existing entries ...
    "your-org/pretrain-dataset": CONTRASTIVE_PROCESSORS,
    "your-org/finetune-dataset": CONTRASTIVE_PROCESSORS,
}
```

Then use them directly:

```bash
# Stage 1: Contrastive Pretraining
python -m tevatron.driver.train \
    --model_name_or_path bert-base-uncased \
    --dataset_name your-org/pretrain-dataset \
    --train_n_passages 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --temperature 0.05 \
    --negatives_x_device \
    --output_dir ./pretrain_checkpoint \
    --num_train_epochs 3

# Stage 2: Finetuning
python -m tevatron.driver.train \
    --model_name_or_path ./pretrain_checkpoint \
    --dataset_name your-org/finetune-dataset \
    --train_n_passages 9 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-6 \
    --temperature 0.01 \
    --negatives_x_device \
    --output_dir ./final_model \
    --num_train_epochs 3
```

### Method 2: Using Custom Dataset with JSON Files

If using local JSON files:

```bash
# Stage 1: Contrastive Pretraining
python -m tevatron.driver.train \
    --model_name_or_path bert-base-uncased \
    --dataset_name json \
    --train_path /path/to/pretrain_data.jsonl \
    --train_n_passages 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --temperature 0.05 \
    --negatives_x_device \
    --output_dir ./pretrain_checkpoint \
    --num_train_epochs 3

# Stage 2: Finetuning
python -m tevatron.driver.train \
    --model_name_or_path ./pretrain_checkpoint \
    --dataset_name json \
    --train_path /path/to/finetune_data.jsonl \
    --train_n_passages 9 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-6 \
    --temperature 0.01 \
    --negatives_x_device \
    --output_dir ./final_model \
    --num_train_epochs 3
```

Note: For JSON datasets, you need to manually set the preprocessor in the code since `PROCESSOR_INFO["json"]` is set to `[None, None, None]`.

## Key Parameters Explained

### Stage 1 (Contrastive Pretraining):
- `--train_n_passages 1`: Only use positive passages; negatives come from in-batch samples
- `--temperature 0.05`: Higher temperature (e.g., 0.05) for softer contrastive learning
- `--per_device_train_batch_size 32`: Larger batch size = more in-batch negatives
- `--negatives_x_device`: Gather negatives across all GPUs (crucial for multi-GPU training)

### Stage 2 (Finetuning):
- `--train_n_passages 9`: Use 1 positive + 8 hard negatives per query
- `--temperature 0.01`: Lower temperature for harder discrimination
- `--learning_rate 5e-6`: Lower learning rate to avoid catastrophic forgetting
- `--negatives_x_device`: Still useful for additional in-batch negatives

## How It Works

### In-Batch Negatives (Stage 1)

With `train_n_passages=1` and batch size 32:

```
Batch contains:
- Query 0 → Passage 0 (positive)
- Query 1 → Passage 1 (positive)
- ...
- Query 31 → Passage 31 (positive)

Score matrix shape: (32, 32)
- Query 0 scores against all 32 passages
- Passage 0 is positive, passages 1-31 are negatives
- Same for all other queries
```

### Hard Negatives (Stage 2)

With `train_n_passages=9` and batch size 32:

```
Batch contains:
- Query 0 → [Pos 0, Neg 0-1, ..., Neg 0-7]
- Query 1 → [Pos 1, Neg 1-1, ..., Neg 1-7]
- ...
- Query 31 → [Pos 31, Neg 31-1, ..., Neg 31-7]

Score matrix shape: (32, 288)  # 32 * 9
- Query 0 has 8 hard negatives + 280 in-batch negatives
- More challenging training signal
```

## Advanced: Custom Preprocessor in Code

If you need more control, modify your training script:

```python
from tevatron.datasets.dataset import HFTrainDataset
from tevatron.datasets.preprocessor import ContrastivePreProcessor

# Override the preprocessor
train_dataset = HFTrainDataset(tokenizer, data_args, cache_dir)
train_dataset.preprocessor = ContrastivePreProcessor(
    tokenizer,
    query_max_length=32,
    text_max_length=256,
    separator=" ",
    q_prefix="query: ",
    p_prefix="passage: "
)
processed_dataset = train_dataset.process()
```

## Tips for Best Results

1. **Batch Size**: Larger batch size in Stage 1 provides more negatives (32-128 recommended)
2. **Temperature**: 
   - Stage 1: 0.05 (softer, helps with initial learning)
   - Stage 2: 0.01-0.02 (harder, better discrimination)
3. **Learning Rate**:
   - Stage 1: 1e-5 to 5e-5
   - Stage 2: 1e-6 to 5e-6 (lower to avoid forgetting)
4. **Epochs**:
   - Stage 1: 3-5 epochs
   - Stage 2: 1-3 epochs
5. **Hard Negatives**: Use BM25 or other retrieval methods to mine challenging negatives for Stage 2

## Differences from TrainPreProcessor

The new `ContrastivePreProcessor` differs from `TrainPreProcessor` in one key way:

- **TrainPreProcessor**: Requires `negative_passages` field (fails without it)
- **ContrastivePreProcessor**: Makes `negative_passages` optional (returns empty list if missing)

This flexibility allows the same preprocessor to handle both training stages.

## Example Training Scripts

### Complete Two-Stage Training Pipeline

```bash
#!/bin/bash

# Stage 1: Contrastive Pretraining
echo "Starting Stage 1: Contrastive Pretraining..."
python -m tevatron.driver.train \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --dataset_name your-org/pretrain-pairs \
    --output_dir ./checkpoints/stage1_pretrain \
    --train_n_passages 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --temperature 0.05 \
    --negatives_x_device \
    --fp16 \
    --dataloader_num_workers 4 \
    --save_strategy epoch \
    --logging_steps 100

# Stage 2: Finetuning with Hard Negatives
echo "Starting Stage 2: Finetuning..."
python -m tevatron.driver.train \
    --model_name_or_path ./checkpoints/stage1_pretrain \
    --dataset_name your-org/finetune-triplets \
    --output_dir ./checkpoints/stage2_finetune \
    --train_n_passages 9 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --temperature 0.01 \
    --negatives_x_device \
    --fp16 \
    --dataloader_num_workers 4 \
    --save_strategy epoch \
    --logging_steps 100

echo "Training complete! Final model saved to ./checkpoints/stage2_finetune"
```

## Troubleshooting

### Issue: "KeyError: 'negative_passages'"
**Solution**: You're using `TrainPreProcessor` instead of `ContrastivePreProcessor`. Either:
1. Add your dataset to `PROCESSOR_INFO` with `CONTRASTIVE_PROCESSORS`
2. Or manually set the preprocessor in code

### Issue: Model doesn't learn in Stage 1
**Solution**: 
- Increase batch size (more in-batch negatives)
- Enable `--negatives_x_device` for multi-GPU training
- Adjust temperature (try 0.05-0.1)

### Issue: Performance drop in Stage 2
**Solution**:
- Lower learning rate (try 1e-6)
- Reduce number of epochs (1-2 may be enough)
- Ensure hard negatives are challenging but not too hard

## Performance Expectations

Typical improvements with two-stage training:

- **Stage 1 only**: 60-70% recall@100 on BEIR datasets
- **Stage 1 + Stage 2**: 70-80% recall@100 on BEIR datasets

The exact improvement depends on:
- Quality of pretraining data
- Quality of hard negatives in Stage 2
- Model architecture and size
- Hyperparameter tuning
