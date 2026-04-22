#!/bin/bash
# Run once on a fresh RunPod pod before anything else.
# Usage: bash setup.sh

set -e

pip install transformers accelerate datasets bitsandbytes \
            rich safetensors wandb tokenizers tqdm gradio Pillow huggingface_hub

# Cache HuggingFace models/datasets to persistent volume
export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Single-GPU default config (no prompts)
accelerate config default
