#!/bin/bash
# Two-stage training: loads a stage-1 LoRA adapter for continued fine-tuning.
# Usage: ./train_sarvam_stage2.sh <stage1_adapter_path>
#   e.g. ./train_sarvam_stage2.sh output/20260301120000

if [ -z "$1" ]; then
    echo "Usage: $0 <stage1_adapter_path>"
    echo "  e.g. $0 output/20260301120000"
    exit 1
fi

STAGE1_ADAPTER="$1"
formatted_time=$(date +"%Y%m%d%H%M%S")
echo "Stage 2 run: $formatted_time (loading adapter from $STAGE1_ADAPTER)"

accelerate launch --config_file ./configs/ddp_config.yaml train.py \
--output_dir output/$formatted_time/ \
--model_name_or_path ../pretrained/sarvam-1/ \
--adapter_path "$STAGE1_ADAPTER" \
--temperature 0.05 \
--train_data_path ../data/processed_shaiva \
--learning_rate 2e-5 \
--per_device_train_batch_size 7 \
--bf16 \
--gradient_accumulation_steps 1 \
--warmup_steps 50 \
--max_steps 500 \
--weight_decay 1e-4 \
--lr_scheduler_type "cosine" \
--save_strategy steps --save_steps 250 --seed 7 \
--remove_unused_columns False \
--log_level info --logging_strategy steps --logging_steps 10 --report_to wandb \
--run_name "${formatted_time}-stage2"
