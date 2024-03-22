
unset CUDA_VISIBLE_DEVICES

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3  && python finetune.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
#     --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat.json' \
#     --output_dir './decompile_llvm_ir_alphaca_lora_seq_len_4k' \
#     --batch_size 64 \
#     --micro_batch_size 16 \
#     --num_epochs 1 \
#     --learning_rate 1e-4 \
#     --cutoff_len 6144 \
#     --val_set_size 10240 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project "decompilation_llvm_ir_lora" \
#     --wandb_watch "full"  \
#     --data_start 0 \
#     --data_end 102400 \


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=0,1  && accelerate launch --config_file ./config.yaml finetune.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json' \
    --output_dir './decompile_llvm_ir_alphaca_lora_seq_len_4k' \
    --resume_from_checkpoint "/data0/xiachunwei/Projects/alpaca-lora/./decompile_llvm_ir_alphaca_lora_full_dataset/checkpoint-6500" \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_epochs 1 \
    --learning_rate 4e-5 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_lora" \
    --wandb_watch "full" --data_end 10240 > >(tee -a lorastdout.log) 2> >(tee -a lorastderr.log >&2)
