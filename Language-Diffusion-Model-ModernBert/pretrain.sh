accelerate launch pretrain.py \
    --experiment_name "LDM_Pretraining_large_dataset" \
    --working_directory "work_dir" \
    --hf_model_name "answerdotai/ModernBert-base"\
    --path_to_prepped_data \
    --num_training_steps 100000 \
    --per_gpu_batch_size 64 \
    --gradient_accumulation_steps 4\