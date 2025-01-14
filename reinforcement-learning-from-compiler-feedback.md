
>> Train with RL:
```shell
export CUDA_VISIBLE_DEVICES=2 && python3 models/llmcompiler/rl_exebench.py     --model=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd     --tokenizer_name=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd     --reward_model_name=/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd     --input_max_length=512 --output_max_length=2048 --batch_size=16     --mini_batch_size=4     --ppo_epochs=2     --log_with=wandb 
```

>> After traning with LoRA, we first merge the LoRA weights with the base model:
```shell
python3 utils/merge_peft_weights_to_llama.py \
    --base_model /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd  \
    --merged_model /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd-rl-ppo-reward-length-step-20 \
    --peft_weight_dir /home/xiachunwei/Projects/checkpoints-decompilation/rl-ppo-reward-length-exebench-split1/rl-ppo-reward-lengthstep_20
```

>> Then inference with llm:

```shell
export CUDA_VISIBLE_DEVICES=3 && python3 models/llmcompiler/llmcompiler_generate.py     --batch_size 32 --num_beams 1     --pretrained_model_path "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"     --dataset_path "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"     --result_file exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-step-20-bs-32-beams-1 --lora_adapter_path /home/xiachunwei/Projects/checkpoints-decompilation/rl-ppo-reward-length-exebench-split1/rl-ppo-reward-lengthstep_20

export CUDA_VISIBLE_DEVICES=3 && python3 models/llmcompiler/llmcompiler_generate.py     --batch_size 32 --num_beams 1     --pretrained_model_path "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"     --dataset_path "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_4_llvm_extract_func_ir_assembly_O2"     --result_file exebench_train_synth_rich_io_filtered_llvm_ir_4_llm-compiler-13b-ftd-rl-ppo-length_reward_pack_similar_length_samples-step-60-bs-32-beams-1 --lora_adapter_path /home/xiachunwei/Projects/checkpoints-decompilation/runs_rl_length_reward_pack_similar_length_samples/step_60
```

>> After that, validate the generated results by compiling and runing them:

```shell
python3 utils/evaluate_exebench.py --path_to_json exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-compilable0.1--step-40-bs-32-beams-1.json  --path_to_dataset /home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2  --path_to_result exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-compilable0.1--step-40-bs-32-beams-1_validate_exebench.json

python3 utils/evaluate_exebench.py --path_to_json exebench_train_synth_rich_io_filtered_llvm_ir_4_llm-compiler-13b-ftd-rl-ppo-length_reward_pack_similar_length_samples-step-60-bs-32-beams-1.json  --path_to_dataset /home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_4_llvm_extract_func_ir_assembly_O2  --path_to_result exebench_train_synth_rich_io_filtered_llvm_ir_4_llm-compiler-13b-ftd-rl-ppo-length_reward_pack_similar_length_samples-step-40-bs-32-beams-1_validate_exebench.json --validation_dir=`pwd`/tmp_validate_exebench_split_4
```


>> Draw the distribution of IO-accuracy/Compilable/Error distribution according to the seq length:

```shell
python3 utils/draw_exebench_distribution.py \
    --data_files exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8_validate_exebench.json \
    --pretrained_model_path /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --fig_file_path exebench_distribution-train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8.png

python3 utils/draw_exebench_distribution.py \
    --data_files exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-step-40-bs-1-beams-8.json \
    --pretrained_model_path /home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd \
    --fig_file_path exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-step-40-bs-1-beams-8.png
```

```shell
export CUDA_VISIBLE_DEVICES=1 && python3 models/llmcompiler/llmcompiler_generate.py     --batch_size 32 --num_beams 1     --pretrained_model_path "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"     --dataset_path "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"     --result_file exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-reward-length-compilable0.1--step-40-bs-32-beams-1 --lora_adapter_path /home/xiachunwei/Projects/checkpoints-decompilation/runs_rl_compilable/runs_rl_compilablestep_40
```

>> Analyze the error type

After executing `evaluate_exebench`, for each record, one directory will be generated.
Within each there will be four files: 
- `target.ll`: The ground truth LLVM IR
- `target.s`: the compiled assembly generated from `target.ll`
- `predict.ll`: the LLVM IR generated by the LLM
- `predict.s` or `errory_type`: if the LLVM IR is semantic right, then the `predict.s` will be generated; otherwise the `error_predict.error` will be generated.

Note, first modify the evaluation dir path and output file path
```shell
bash analysis/get_all_error_predict.sh
```

Analyze the error type using python:
```shell
python3 analysis/analyze_error_type.py
```
This will produce the `analysis/error_type_pie.png` which shows how much percent for each error type.