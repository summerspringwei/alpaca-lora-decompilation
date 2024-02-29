#!/bin/bash

# Run the server in generate.py
# export CUDA_VISIBLE_DEVICES=0 && python3 run_generate.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
#     --lora_weights './decompile_alphaca_lora' \
#     --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample128.json' \
#     --result_file "./mayval/result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample128.json" | tee tmp_val.log


# export CUDA_VISIBLE_DEVICES=3 && python3 run_generate.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
#     --lora_weights './decompile_alphaca_lora' \
#     --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K.json' \
#     --result_file "./mayval/result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K_our_lora.json" | tee ./mayval/val_tail1K_our_lora.log


# export CUDA_VISIBLE_DEVICES=2 && python3 run_generate.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-Instruct-hf' \
#     --lora_weights '' \
#     --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K.json' \
#     --result_file "./mayval/result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K.json" | tee codellama_tail1k.log

export CUDA_VISIBLE_DEVICES=1 && python3 run_generate.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --lora_weights './decompile_llvm_ir_alphaca_lora/checkpoint-1800' \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat.json' \
    --result_file "./decompilation_val/result_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_shuffle_0_100.json" \
    --start_idx 0 \
    --end_idx 100 | tee ./decompilation_val/val_llvm-ir-llc-0-10.log
