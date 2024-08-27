
# unset CUDA_VISIBLE_DEVICES

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch --config_file ./accelerate_config.yaml \
#     finetune_full_starcode2.py \
#     --base_model '/data0/xiachunwei/Dataset/deepseek-coder-1.3b-base' \
#     --data_path "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json" \
#     --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2' \
#     --batch_size 128 \
#     --micro_batch_size 16 \
#     --num_epochs 1 \
#     --learning_rate 1e-4 \
#     --cutoff_len 4224 \
#     --val_set_size 128 \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b" \
#     --wandb_watch "full"  \
#     --data_start 0 \
#     --num_proc 32 \
#     --data_end 1024 > >(tee -a finetune_full_stdout.log) 2> >(tee -a finetune_full_stderr.log >&2)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch --config_file ./accelerate_config.yaml \
    finetune_full_starcode2.py \
    --base_model '/data0/xiachunwei/Dataset/deepseek-coder-1.3b-base' \
    --data_path "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4224 \
    --val_set_size 40960 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b" \
    --resume_from_resume_from_checkpoint ./decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2/tmp-checkpoint-100/
    --wandb_watch "full"  \
    --num_proc 40  > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)

unset CUDA_VISIBLE_DEVICES

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch --config_file ./accelerate_config.yaml \
    finetune_full_starcode2.py \
    --base_model '/data0/xiachunwei/Dataset/deepseek-coder-1.3b-base' \
    --data_path "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2-scratch' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4224 \
    --val_set_size 40960 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b" \
    --wandb_watch "full"  \
    --num_proc 40  > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch --config_file ./accelerate_config.yaml \
    finetune_full_starcode2.py \
    --base_model '/data/xiachunwei/Datasets/Models/deepseek-coder-1.3b-base' \
    --data_path "/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2-average-2_chat.json" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2-scratch-test' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4224 \
    --val_set_size 1024 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b-test" \
    --wandb_watch "full"  \
    --num_proc 40 --data_end=10240 > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=2,3 && accelerate launch --config_file ./accelerate_config.yaml \
    finetune_full_starcode2.py \
    --base_model '/data0/xiachunwei/Dataset/deepseek-coder-1.3b-base' \
    --data_path "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2-scratch' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4224 \
    --val_set_size 40960 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b" \
    --wandb_watch "full"  \
    --num_proc 40  > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --config_file ./accelerate_config.yaml \
    finetune_full_starcode2.py \
    --base_model '/data/xiachunwei/Datasets/Models/deepseek-coder-6.7b-base/' \
    --data_path "/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2-average-2_chat.json" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-6.7b-base_seq_len_4k-bbcount-2-scratch' \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4224 \
    --val_set_size 1024 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-6.7b" \
    --wandb_watch "full"  \
    --resume_from_resume_from_checkpoint ./decompile_llvm_ir_alphaca_deepseek-coder-6.7b-base_seq_len_4k-bbcount-2-scratch/checkpoint-200 \
    --num_proc 40 > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  && export CUDA_VISIBLE_DEVICES=3 && python3 starcode2_finetune.py \
    --base_model '/data/xiachunwei/Datasets/Models/deepseek-coder-1.3b-base' \
    --data_path "/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort" \
    --output_dir './decompile_llvm_ir_alphaca_deepseek-coder-1.3b-filtered' \
    --batch_size 64 \
    --micro_batch_size 64 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 4096 \
    --val_set_size 1024 \
    --train_on_inputs \
    --group_by_length \
    --wandb_project "decompilation_llvm_ir_deepseek-coder-1.3b" \
    --wandb_watch "full"  \
    --data_end 10240 \
    --num_proc 40 > >(tee finetune_full_stdout.log) 2> >(tee finetune_full_stderr.log >&2)
