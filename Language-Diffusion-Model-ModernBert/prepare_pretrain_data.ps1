python prepare_pretrain_data.py `
    --test_split_pct 0.005 `
    --context_length 1024 `
    --path_to_data_store "C:\Users\5226255\OneDrive - Lowe's Companies Inc\Desktop\Diffusion\Language-Diffusion-Model-ModernBert\modernbert_dataset" `
    --huggingface_cache_dir "C:\Users\5226255\OneDrive - Lowe's Companies Inc\Desktop\Diffusion\Language-Diffusion-Model-ModernBert\hf_cache" `
    --dataset_split_seed 42 `
    --num_workers 4 `
    --hf_model_name "answerdotai/ModernBERT-base"