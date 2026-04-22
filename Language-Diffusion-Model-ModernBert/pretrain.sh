#!/bin/bash
# Pretrain the diffusion LM on ModernBERT-base.
# Run prepare_pretrain_data.sh first.
# Usage: bash pretrain.sh

accelerate launch --mixed_precision bf16 pretrain.py \
    --experiment_name "LDM_pretrain_base" \
    --working_directory "/workspace/experiments" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --path_to_prepped_data "/workspace/data/pretrain" \
    --num_training_steps 100000 \
    --per_gpu_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 2000 \
    --evaluation_interval 2500 \
    --checkpoint_interval 5000 \
    --num_workers 4 \
    --log_wandb
