# Installation Guide

This repository contains two Python packages that need to be installed in editable mode:
- **llm2vec**: Located at `llm2vec/`
- **tevatron-tevatron-v1**: Located at `tevatron-tevatron-v1/`

## Installation Steps

### 1. Install Dependencies

First, install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages including:
- PyTorch and related CUDA libraries
- Transformers, datasets, and other HuggingFace libraries
- Deep learning utilities (peft, accelerate, deepspeed, etc.)
- Evaluation and analysis tools

### 2. Install Packages in Editable Mode (Without Dependencies)

After installing the dependencies, install both packages in editable mode **without** reinstalling their dependencies:

```bash
# Install llm2vec in editable mode
cd llm2vec
pip install -e . --no-deps
cd ..

# Install tevatron-tevatron-v1 in editable mode
cd tevatron-tevatron-v1
pip install -e . --no-deps
cd ..
```

**Why use `--no-deps`?**
- The `--no-deps` flag prevents pip from installing the dependencies listed in each package's `setup.py`
- This avoids redundant installations and potential version conflicts
- Since all dependencies are already installed from `requirements.txt`, we only need the packages themselves

### 3. Verify Installation

You can verify that both packages are installed correctly:

```bash
pip list | grep -E "llm2vec|tevatron"
```

You should see both packages listed with their local paths, indicating they're installed in editable mode.

## What is Editable Mode?

Installing in editable mode (`-e` or `--editable`) means:
- The packages are linked to their source directories rather than copied
- Any changes you make to the source code are immediately reflected without reinstalling
- Perfect for development and experimentation

## Package Dependencies

### llm2vec
The `llm2vec` package originally requires:
- numpy, tqdm, torch, peft, datasets, evaluate, scikit-learn

### tevatron-tevatron-v1
The `tevatron-tevatron-v1` package originally requires:
- transformers>=4.10.0, datasets>=1.1.3

All these dependencies (and more) are already covered by the `requirements.txt` file.

## Troubleshooting

If you encounter any issues:

1. **Import errors**: Ensure you've installed all dependencies from `requirements.txt` first
2. **Version conflicts**: Check that the versions in `requirements.txt` are compatible with your system
3. **CUDA issues**: Make sure you have the appropriate CUDA version installed for the PyTorch build

## Alternative Installation (Not Recommended)

If you want to install with automatic dependency resolution (may cause version conflicts):

```bash
# This will install packages WITH their dependencies
pip install -e llm2vec/
pip install -e tevatron-tevatron-v1/
```

However, this is **not recommended** as it may override versions specified in `requirements.txt`.
