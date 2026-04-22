#!/bin/bash
# Prepare SFT data from Open-Orca.
# Requires pretrain data prep to have run first (shares the same tokenizer setup).
# Usage: bash prepare_sft_data.sh

python prepare_sft_data.py \
    --test_split_pct 0.01 \
    --context_length 1024 \
    --path_to_data_store /workspace/data/sft \
    --huggingface_cache_dir /workspace/hf_cache \
    --dataset_split_seed 42 \
    --num_workers 8 \
    --hf_model_name "answerdotai/ModernBERT-base"
