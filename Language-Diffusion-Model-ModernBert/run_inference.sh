#!/bin/bash
PROMPT=${1:-"Who was the first president of the United States?"}

python inference.py \
    --safetensors_path "/workspace/experiments/LDM_sft_openorca/final_model/model.safetensors" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --seq_len 64 \
    --num_steps 64 \
    --strategy low_confidence \
    --prompt "$PROMPT"
