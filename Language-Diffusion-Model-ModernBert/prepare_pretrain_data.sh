#!/bin/bash
# Prepare pretraining data.
# Default: Project Gutenberg (small, fast).
# Add --large_dataset for FineWeb + FineWeb-Edu + Wikipedia (~20B tokens).
# Usage: bash prepare_pretrain_data.sh

python prepare_pretrain_data.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --path_to_data_store /workspace/data/pretrain \
    --huggingface_cache_dir /workspace/hf_cache \
    --dataset_split_seed 42 \
    --num_workers 8 \
    --hf_model_name "answerdotai/ModernBERT-base"
    # --large_dataset   # uncomment for FineWeb + FineWeb-Edu + Wikipedia
