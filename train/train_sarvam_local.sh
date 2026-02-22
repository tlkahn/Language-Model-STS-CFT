formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

# Let MPS allocate memory more freely (avoids premature OOM)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use `python` directly — HF Trainer auto-detects MPS.
# `accelerate launch` with the local config defaults to CPU when CUDA is absent.
python train.py \
--output_dir output/$formatted_time/ \
--model_name_or_path ../pretrained/sarvam-1/ \
--temperature 0.05 \
--train_data_path ../data/processed_pilot \
--learning_rate 5e-5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--warmup_steps 100 \
--max_steps 1000 \
--weight_decay 1e-4 \
--lr_scheduler_type "cosine" \
--lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
--save_strategy steps --save_steps 500 --seed 7 \
--remove_unused_columns False \
--log_level info --logging_strategy steps --logging_steps 10 \
--dataloader_num_workers 0 \
--report_to wandb --run_name $formatted_time
