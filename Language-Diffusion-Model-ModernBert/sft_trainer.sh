#!/bin/bash
# SFT fine-tune on Open-Orca starting from a pretrained checkpoint.
# Run pretrain.sh first, then prepare_sft_data.sh.
# Usage: bash sft_trainer.sh

accelerate launch --mixed_precision bf16 sft_trainer.py \
    --experiment_name "LDM_sft_openorca" \
    --working_directory "/workspace/experiments" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --path_to_pretrained_checkpoint "/workspace/experiments/LDM_pretrain_base/final_model/model.safetensors" \
    --path_to_prepped_data "/workspace/data/sft" \
    --num_training_steps 30000 \
    --per_gpu_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --evaluation_interval 2500 \
    --checkpoint_interval 10000 \
    --num_workers 4 \
    --log_wandb
