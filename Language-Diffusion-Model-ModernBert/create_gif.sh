#!/bin/bash
# Generate inference GIF from SFT model.
# Usage: bash create_gif.sh

# Conditional (Q&A) with SFT model
python create_gif.py \
    --safetensors_path "/workspace/experiments/LDM_sft_openorca/final_model/model.safetensors" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --prompt "What is artificial intelligence?" \
    --seq_len 256 \
    --num_steps 128 \
    --strategy low_confidence \
    --frame_every 4 \
    --fps 8 \
    --output inference_sft.gif

# Unconditional with pretrain model (may produce better text)
python create_gif.py \
    --safetensors_path "/workspace/experiments/LDM_pretrain_base/final_model/model.safetensors" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --seq_len 128 \
    --num_steps 128 \
    --strategy low_confidence \
    --frame_every 4 \
    --fps 8 \
    --output inference_pretrain.gif
