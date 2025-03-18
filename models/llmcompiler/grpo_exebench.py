import torch
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import LlamaTokenizer
from peft import LoraConfig
from torch.profiler import profile, record_function, ProfilerActivity
import fire
import wandb

from utils.extract_code import extract_llmcompiler_code_blocks
from utils.evaluate_exebench import validate_by_execution
from models.llmcompiler.processing import pack_similar_length_samples
from utils.preprocessing_assembly import preprocessing_assembly

def build_dataset(path_to_dataset, tokenizer, max_input_length=512, batch_size=8, num_chunks=2):
    exebench_dataset = load_from_disk(path_to_dataset)
    
    def copy_asm_to_prompt(example):
        example["prompt"] = "[ INST ] Disassemble this code to LLVM-IR :\n<code> " + preprocessing_assembly(example["asm"]["code"][-1]) + "</code> [/INST]"

        return example

    formatted_dataset = exebench_dataset.map(copy_asm_to_prompt)
    formatted_dataset = formatted_dataset.filter(lambda x: len(tokenizer(x['prompt'])["input_ids"]) < max_input_length)
    # exebench_dataset = pack_similar_length_samples(formatted_dataset, batch_size, num_chunks, sort_key_function=lambda x: len(tokenizer(x['prompt'])["input_ids"]))

    return formatted_dataset


def main(
        path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100",
        model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-7b-ftd",
        max_completion_length=4096,
        num_generations=8,
        save_steps=50,
        batch_size=8,
        validation_dir = "validation/tmp_rl_validation"):

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    exebench_dataset = build_dataset(path_to_dataset, tokenizer)

    for record in exebench_dataset:
        print(len(tokenizer(record['prompt'])["input_ids"]))

    training_args = GRPOConfig(output_dir="llmcompiler-7b-GRPO", 
                                logging_steps=1,
                                max_completion_length=max_completion_length,
                                num_generations=num_generations,
                                save_steps=save_steps
                                )

    training_args.set_training(batch_size=batch_size, 
                            num_epochs=1, 
                            # max_steps=1, 
                            gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


    def reward_compilation(completions, **kwargs):
        # Convert kwargs to a list to dict to match the batch size
        original_input = [{} for _ in range(len(completions))]
        predict_list_length = []
        for k, v in kwargs.items():
            for i in range(len(v)):
                original_input[i][k] = v[i]
        validation_list = []
        for row, completion in zip(original_input, completions):
            print(len(tokenizer(completion)["input_ids"]))
            match_results = extract_llmcompiler_code_blocks(completion)
            predict_ir = "Failed to extract IR"
            if len(match_results) > 0:
                predict_ir = match_results[0]
            record = {
                'file': row["path"],
                'predict': [predict_ir],
                'output': row["llvm_ir"]["code"][-1]
            }
            record = validate_by_execution(record, row, validation_dir)
            validation_list.append(record)
            predict_list_length.append(len(tokenizer(predict_ir)["input_ids"]))
        predict_reward = [1 if r["predict_compile_success"][0] is True else 0 for r in validation_list]
        executable_reward = [1 if r["predict_execution_success"][0] is True else 0 for r in validation_list]
        wandb.log({
            "results": 
                wandb.Table(columns=["compile", "executable"], data = [[p, e] for p, e in zip(predict_reward, executable_reward)])
        })
        return predict_reward


    trainer = GRPOTrainer(
        model=model_path,
        processing_class=tokenizer,
        reward_funcs=reward_compilation,
        args=training_args,
        train_dataset=exebench_dataset,
        peft_config = lora_config,
    )

    # torch.cuda.memory._record_memory_history()
    trainer.train()
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


if __name__ == "__main__":
    fire.Fire(main)
