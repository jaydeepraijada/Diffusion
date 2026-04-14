python inference.py \
    --safetensors_path "work_dir/LDM_Pretraining_large_ft_alpaca/final_model/model.safetensors" \
    --seq_len 512 \
    --num_steps 512 \
    --hf_model_name answerdotai/ModernBERT-large \
    --prompt "What is artificial intelligence"