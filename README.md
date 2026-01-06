## Overview
This repo is the official implementation repo of our technical report ***LACONOC: Dense-Level Effectiveness for Scalable Sparse Retrieval via a Two-Phase Training Curriculum***. You can read the writeup here:
[arXiv](https://arxiv.org/abs/2601.01684). We provide a comprehensive framework designed for training, evaluating, and deploying large language models (LLMs) as sparse retrieval systems. It includes tools for:
- Training and fine-tuning models
- Sparse encoding and retrieval
- Experiment tracking and evaluation

This repository contains two Python packages:
- **llm2vec**: A package for vector-based language model experiments.
- **tevatron-tevatron-v1**: A package for sparse retrieval and indexing.

## Features
- Support for HuggingFace models and datasets
- Integration with Weights & Biases for experiment tracking
- Accelerated training with DeepSpeed and Accelerate
- Predefined configurations for various training and evaluation workflows
- Editable installation for development

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8) and a CUDA-compatible GPU for training. The following steps demonstrate using Conda and pip. But we recommend using uv for pain-free setup. 

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the packages in editable mode:
   ```bash
   # Install llm2vec
   cd llm2vec
   pip install -e . --no-deps
   cd ..

   # Install tevatron-tevatron-v1
   cd tevatron-tevatron-v1
   pip install -e . --no-deps
   cd ..
   ```

4. Verify installation:
   ```bash
   pip list | grep -E "llm2vec|tevatron"
   ```

For detailed installation instructions, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

## Environment Setup

Set up the required environment variables for HuggingFace and Weights & Biases:

```bash
# HuggingFace Authentication
export HF_TOKEN="your_huggingface_token_here"

# Weights & Biases (Optional)
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="sparse-project"
export WANDB_RUN_NAME="experiment-001"
```

For more details, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

## Repository Structure

```
.
├── accelerate_configs/       # Predefined configurations for Accelerate
├── docs/                     # Documentation files
├── eval_logs/                # Logs from evaluation runs
├── llm2vec/                  # llm2vec package source code
├── output/                   # Output files and results
├── splade_scripts/           # Scripts for SPLADE training and encoding
├── src/                      # Source code for training and retrieval
├── tevatron-tevatron-v1/     # tevatron package source code
├── test_configs/             # Test configurations
├── train_configs/            # Training configurations
├── train_scripts/            # Training scripts
└── wandb/                    # Weights & Biases logs
```

## Usage

### Training
To train a model using SPLADE:
```bash
python src/train_splade_causal.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --output_dir "outputs/run1" \
    # Add other arguments as needed
```

### Encoding
To encode data using SPLADE:
```bash
python src/encode_splade_causal.py \
    --model_name_or_path "outputs/run1" \
    --input_file "data/input.json" \
    --output_file "data/encoded.json"
```

### Retrieval
To perform retrieval:
```bash
python src/retrieve_from_files_pyseismic.py \
    --index_file "data/index.json" \
    --query_file "data/queries.json" \
    --output_file "data/results.json"
```

## Artifacts

We provide our trained models for download on HuggingFace:
- [LACONIC-1B](https://huggingface.co/utahnlp/laconic-1b)
- [LACONIC-3B](https://huggingface.co/utahnlp/laconic-3b)
- [LACONIC-8B](https://huggingface.co/utahnlp/laconic-8b)

For dataset used in the experiments:
- [Pretraining](https://huggingface.co/datasets/utahnlp/nomic-embed-pretrain-lite)
- [Finetuning](https://huggingface.co/datasets/rlhn/rlhn-680K)

## Citation

If you use this project in your research or use the trained models, please cite our work:

```bibtex
@misc{xu2026laconicdenseleveleffectivenessscalable,
      title={LACONIC: Dense-Level Effectiveness for Scalable Sparse Retrieval via a Two-Phase Training Curriculum}, 
      author={Zhichao Xu and Shengyao Zhuang and Crystina Zhang and Xueguang Ma and Yijun Tian and Maitrey Mehta and Jimmy Lin and Vivek Srikumar},
      year={2026},
      eprint={2601.01684},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2601.01684}, 
}
```

## License

TODO

---
For more information, see the documentation in the `docs/` folder.
