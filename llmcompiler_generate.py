
import os
import json
from typing import List, Dict

import torch
import transformers
from tqdm import tqdm
from utils.prompter import Prompter
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset, load_from_disk
from safetensors import safe_open
import logging
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_llmcompiler_model(pretrained_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    model.to(device)
    model.eval()

    return model, tokenizer

input_str = "Disassemble this code to LLVM-IR"

def evaluate_batch(
        prompter, tokenizer, model,
        instruction_list: List[str],
        input_list: List[str]=None,
        temperature=0.1,
        top_p=0.95,
        top_k=10,
        num_beams=1,
        max_new_tokens=1024*2,
        **kwargs,
    ):
        """Currently we only support generate one candidate for each input."""
        prompt_list = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instruction_list, input_list)]
        inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024*4,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
            s = generation_output.sequences
            outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
            results = [prompter.get_response(o) for o in outputs]
        return results


def exebench_evaluate(
    programs,
    prompter, tokenizer, model,
    result_file: str = "val_result.json",
    batch_size: int = 1
    ):
    results = []
    count = len(programs) // batch_size
    with open(result_file+"_inc", "a") as f:
        for i in range(count):
            # try:
                start = i * batch_size
                end = min((i+1) * batch_size, len(programs))
                batch_p = [input_str] * batch_size
                batch_input = [p["asm"]["code"][-1] for p in programs.select(range(start, end))]
                batch_predict = evaluate_batch(prompter, tokenizer, model, batch_p, input_list=batch_input)
                batch_output = [
                    {"instruction":input_str, 
                     "input":p["asm"]["target"][-1], "predict": predict, "file": p["path"], "output": p["llvm_ir"]["code"]}
                    for p, predict in zip(programs.select(range(start, end)), batch_predict)
                ]

                results.extend(batch_output)
                for p in batch_output:
                    json.dump(p, f, indent=4, sort_keys=True, separators=(',', ':'))
                    f.write(",\n")
                    f.flush()
            # except Exception as e:
            #     logging.error(e)
            #     continue
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True, separators=(',', ':'))


def exebench_main():
    pretrained_model_path = model = "/data/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"
    model, tokenizer = get_llmcompiler_model(pretrained_model_path)
    prompter = Prompter("llmcompiler")
    path = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"
    programs = load_from_disk(path)
    exebench_evaluate(programs, prompter, tokenizer, model, result_file="exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd", batch_size=2)


if __name__ == "__main__":
    # fire.Fire(main)
    exebench_main()
