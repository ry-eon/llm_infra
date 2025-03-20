#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=4,5,6 # GPU에 따라 수정

# Model ID
MODEL_ID="Bllossom/llama-3.2-Korean-Bllossom-3B"

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Number of processes to run (matching the number of GPUs)
NUM_PROC=3

# Run the training script using torchrun
torchrun --nproc_per_node=$NUM_PROC train.py \
    --model_id $MODEL_ID \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT