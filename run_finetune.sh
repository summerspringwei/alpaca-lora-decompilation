
# python finetune.py \
#     --base_model 'huggyllama/llama-7b' \
#     --data_path ./alpaca_data_small.json \
#     --output_dir './xcw_alphaca_lora'

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0  && python finetune.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample10K.json' \
    --output_dir './decompile_alphaca_lora' \
    --batch_size 64 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 4224 \
    --val_set_size 28 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_lora" \
    --wandb_watch "full" \
    --val_set_size 1024


unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1 && torchrun --nnodes 1 --nproc-per-node gpu \
    finetune.py \
    --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
    --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_10-100K.json' \
    --num_proc 32 \
    --output_dir './decompile_alphaca_lora' \
    --batch_size 64 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 4224 \
    --val_set_size 1024 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_lora" \
    --wandb_watch "full" \
    --resume_from_checkpoint './decompile_alphaca_lora'

# export CUDA_VISIBLE_DEVICES=0  && python finetune.py \
#     --base_model '/data0/xiachunwei/Dataset/CodeLlama-7b-hf' \
#     --data_path '/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample128.json' \
#     --output_dir './decompile_alphaca_lora' \
#     --batch_size 64 \
#     --micro_batch_size 16 \
#     --num_epochs 1 \
#     --learning_rate 1e-4 \
#     --cutoff_len 4224 \
#     --val_set_size 28 \
#     --lora_r 16 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project "decompilation_lora" \
#     --wandb_watch "full" \
#     --val_set_size 16
