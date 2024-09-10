#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 && \
ncu --metrics dram__bytes_read,gpu__time_duration --clock-control none  -o ncu-rl-ppo-sl2 -f --target-processes all \
python3 models/llmcompiler/rl_exebench.py \
    --model=facebook/llm-compiler-7b-ftd \
    --tokenizer_name=facebook/llm-compiler-7b-ftd \
    --reward_model_name=facebook/llm-compiler-7b-ftd \
    --input_max_length=384 \
    --output_max_length=2 \
    --batch_size=32 \
    --mini_batch_size=16 \
    --ppo_epochs=2 \
    --log_with=wandb


export CUDA_VISIBLE_DEVICES=2 && \
python3 models/llmcompiler/rl_exebench.py \
    --model=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --tokenizer_name=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --reward_model_name=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --input_max_length=384 \
    --output_max_length=1536 \
    --batch_size=32 \
    --mini_batch_size=16 \
    --ppo_epochs=2 \
    --log_with=wandb
