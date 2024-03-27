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



export CUDA_VISIBLE_DEVICES=3 && python3 run_generate.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --lora_weights './decompile_llvm_ir_alphaca_lora_full_dataset/checkpoint-6500' \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat-tail-100.json' \
    --result_file "./decompilation_val/result_bbcount-2_beamsearch-16_6.5K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_tail-100.json" \
    --start_idx 0 \
    --end_idx 100



export CUDA_VISIBLE_DEVICES=1 && python3 run_generate.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --lora_weights './checkpoints_decompilation/decompile_llvm_ir_alphaca_lora_seq_len_4k_with_flashattn_maybe_wrong/checkpoint-4100' \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat-head-100.json' \
    --result_file "./validation_decompilation/result_bbcount-2_beamsearch-16_4.1K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_head-100.json" \
    --start_idx 0 \
    --end_idx 100

export CUDA_VISIBLE_DEVICES=0,1 && accelerate launch --num_processes 2 --config_file ./config.yaml run_generate_accelerate.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --lora_weights './decompile_llvm_ir_alphaca_lora_seq_len_4k_with_flashattn_maybe_wrong/checkpoint-4100' \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat-head-100.json' \
    --result_file "./decompilation_val/result_bbcount-2_beamsearch-16_4.1K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_head-100.json" \
    --start_idx 0 \
    --end_idx 100  > >(tee generatestdout.log) 2> >(tee generatestderr.log >&2)


export CUDA_VISIBLE_DEVICES=1 && python3 run_generate.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora' \
    --lora_weights "" \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat-tail-100.json' \
    --result_file "./decompilation_val/result_bbcount-2_beamsearch-16_4.1K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_tail-100.json" \
    --start_idx 0 \
    --end_idx 100  > >(tee generatestdout.log) 2> >(tee generatestderr.log >&2)


export CUDA_VISIBLE_DEVICES=1 && python3 run_generate_vllm.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora' \
    --val_file '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat-tail-100.json' \
    --result_file "./decompilation_val/result_bbcount-2_beamsearch-16_4.1K_AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat_tail-100.json" \
    --start_idx 23 \
    --end_idx 100  > >(tee generatestdout.log) 2> >(tee generatestderr.log >&2)
