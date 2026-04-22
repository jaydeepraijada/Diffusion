#!/bin/bash
# Run inference after SFT.
# Usage: bash inference.sh

erence.py \
    --safetensors_path "/workspace/experiments/LDM_sft_openorca/final_model/model.safetensors" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --seq_len 512 \
    --num_steps 256 \
    --strategy low_confidence \
    --prompt "What is artificial intelligence"
python inf