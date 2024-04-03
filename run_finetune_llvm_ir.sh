
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


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3  && accelerate launch --config_file ./accelerate_config.yaml finetune.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2-average-2_chat.json' \
    --output_dir './decompile_alphaca_lora_bb2_average_inst2' \
    --resume_from_checkpoint "./decompile_alphaca_lora_bb2_average_inst2/checkpoint-500" \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 4e-5 \
    --cutoff_len 4224 \
    --val_set_size 10240 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_lora" \
    --wandb_watch "full" > >(tee -a finetunestdout.log) 2> >(tee -a finetunestderr.log >&2)


# unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=2,3 && torchrun --nnodes 1 --nproc-per-node gpu \
#     finetune.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
#     --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2-average-2_chat.json' \
#     --num_proc 32 \
#     --output_dir './decompile_alphaca_lora_bb2_average_inst2' \
#     --batch_size 64 \
#     --micro_batch_size 16 \
#     --num_epochs 1 \
#     --learning_rate 1e-4 \
#     --cutoff_len 4224 \
#     --val_set_size 10240 \
#     --lora_r 16 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project "decompilation_lora" \
#     --wandb_watch "full" \
#     --resume_from_checkpoint '/data0/xiachunwei/Projects/checkpoints-decompilation/decompile_llvm_ir_alphaca_lora_seq_len_4k_with_flashattn_maybe_wrong/checkpoint-4100' > >(tee finetunestdout.log) 2> >(tee finetunestderr.log >&2)
