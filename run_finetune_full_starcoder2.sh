
unset CUDA_VISIBLE_DEVICES

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch \
    finetune_full_starcode2.py \
    --base_model '/data0/xiachunwei/Dataset/phi-1_5' \
    --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat.json' \
    --output_dir './decompile_llvm_ir_alphaca_phi-1_5_seq_len_4k' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 4096 \
    --val_set_size 128 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_phi-1_5" \
    --wandb_watch "full"  \
    --data_start 0 \
    --num_proc 4 \
    --data_end 1024 2>&1  | tee finetune_llvm_ir_starcoder2.log 



# unset CUDA_VISIBLE_DEVICES

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2 && python3 \
#     finetune_full_starcode2.py \
#     --base_model '/data0/xiachunwei/Dataset/starcoder2-3b' \
#     --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_chat.json' \
#     --output_dir './decompile_llvm_ir_alphaca_starcoder2_seq_len_4k' \
#     --batch_size 32 \
#     --micro_batch_size 4 \
#     --num_epochs 1 \
#     --learning_rate 1e-4 \
#     --cutoff_len 4096 \
#     --val_set_size 128 \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project "decompilation_llvm_ir_starcoder2" \
#     --wandb_watch "full"  \
#     --data_start 0 \
#     --num_proc 4 \
#     --data_end 1024 2>&1  | tee finetune_llvm_ir_starcoder2.log 


