#!/bin/bash
# Generate inference GIF from SFT model.
# Usage: bash create_gif.sh

python create_gif.py \
    --safetensors_path "/workspace/experiments/LDM_sft_openorca/final_model/model.safetensors" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --prompt "What is artificial intelligence?" \
    --seq_len 256 \
    --num_steps 128 \
    --strategy low_confidence \
    --frame_every 4 \
    --fps 8 \
    --output inference.gif
