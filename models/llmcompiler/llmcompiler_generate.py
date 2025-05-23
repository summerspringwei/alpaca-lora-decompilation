"""Module for evaluating the model on ExeBench dataset using vllm.

Typical useage example:

export CUDA_VISIBLE_DEVICES=2 && python3 models/llmcompiler/llmcompiler_generate.py \
    --batch_size 32 --num_beams 1 \
    --pretrained_model_path "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd-rl-ppo-step-40" \
    --dataset_path "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2" \
    --result_file "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-step-40-bs-32-beams-1" \
    --lora_adapter_path ""
"""
import json
import logging
from typing import List

import fire
import torch
from tqdm import tqdm
from utils.prompter import Prompter
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from utils.preprocessing_assembly import preprocessing_assembly

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
    prompter,
    tokenizer,
    model,
    instruction_list: List[str],
    input_list: List[str] = None,
    temperature=0.1,
    top_p=0.95,
    top_k=10,
    num_beams=1,
    max_new_tokens=1024 * 2,
    **kwargs,
):
    """Currently we only support generate one candidate for each input."""
    prompt_list = [
        prompter.generate_prompt(instruction, input)
        for instruction, input in zip(instruction_list, input_list)
    ]
    inputs = tokenizer(prompt_list,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generation_config = GenerationConfig(do_sample=True,
                                         top_k=top_k,
                                         temperature=temperature,
                                         top_p=top_p,
                                         num_return_sequences=1,
                                         eos_token_id=tokenizer.eos_token_id,
                                         max_length=1024 * 4,
                                         max_new_tokens=max_new_tokens,
                                         **kwargs)
    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           generation_config=generation_config,
                                           return_dict_in_generate=True,
                                           output_scores=True,
                                           early_stopping=True)
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        results = [prompter.get_response(o) for o in outputs]
    return results


def vllm_evaluate_batch(
    prompter,
    llm,
    instruction_list: List[str],
    input_list: List[str] = None,
    temperature=0.1,
    top_p=0.95,
    top_k=10,
    number_of_generation_per_sample=1,
    max_new_tokens=1024 * 4,
    lora_adapter_path=None,
):
    """Currently we only support generate one candidate for each input."""
    prompt_list = [
        prompter.generate_prompt(instruction, input)
        for instruction, input in zip(instruction_list, input_list)
    ]
    sampling_params = SamplingParams(n=number_of_generation_per_sample,
                                    top_k=top_k,
                                        temperature=temperature,
                                        top_p=top_p,
                                        max_tokens=max_new_tokens,
                                        min_tokens=256)
    lora_request = LoRARequest("lora", 1, lora_adapter_path) if lora_adapter_path else None
    sequences = llm.generate(prompt_list, sampling_params, lora_request=lora_request)
    results = [[out.text for out in seq.outputs] for seq in sequences]
    return results


def exebench_evaluate(programs,
                      prompter,
                      tokenizer,
                      model,
                      result_file: str = "val_result.json",
                      batch_size: int = 1,
                      llm=None,
                      number_of_generation_per_sample=1,
                      lora_adapter_path=None,
                      remove_comments=False):
    results = []
    count = len(programs) // batch_size
    with open(result_file + "_inc", "a") as f:
        for i in tqdm(range(count)):
            try:
                start = i * batch_size
                end = min((i + 1) * batch_size, len(programs))
                batch_p = [input_str] * batch_size
                batch_input = [
                    preprocessing_assembly(p["asm"]["code"][-1], remove_comments=remove_comments)
                    for p in programs.select(range(start, end))
                ]
                if llm is not None:
                    batch_predict = vllm_evaluate_batch(prompter,
                                                        llm,
                                                        batch_p,
                                                        input_list=batch_input,
                                                        number_of_generation_per_sample=number_of_generation_per_sample,
                                                        lora_adapter_path=lora_adapter_path)
                else:
                    batch_predict = evaluate_batch(prompter,
                                                   tokenizer,
                                                   model,
                                                   batch_p,
                                                   input_list=batch_input)
                batch_output = [
                    {
                        "instruction": input_str,
                        "input": p["asm"]["code"][-1],
                        "predict": predict,
                        "file": p["path"],
                        "output": p["llvm_ir"]["code"],
                        "func_head_types": p["func_head_types"]
                    } for p, predict in zip(programs.select(range(start, end)),
                                        batch_predict)                        
                ]
                results.extend(batch_output)
                for p in batch_output:
                    json.dump(p,
                              f,
                              indent=4,
                              sort_keys=True,
                              separators=(',', ':'))
                    f.write(",\n")
                    f.flush()
            except Exception as e:
                logging.error(e)
    with open(result_file+".json", "w") as f:
        json.dump(results, f, indent=4, sort_keys=True, separators=(',', ':'))


def exebench_main(batch_size = 8, 
                  number_of_generation_per_sample = 1,
                  pretrained_model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-7b-ftd",
                  dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100",
                  result_file = f"exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-7b-ftd-beams-8",
                  lora_adapter_path = "llmcompiler-7b-GRPO-execution/checkpoint-100/",
                framework="vllm",
                remove_comments=False
    ):
    
    # model, tokenizer = get_llmcompiler_model(pretrained_model_path)
    model, tokenizer = None, None
    prompter = Prompter("llmcompiler")
    # Sort the programs by length
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    programs = load_from_disk(dataset_path)
    # We can consider sort according to number of instructions

    enable_lora=True if lora_adapter_path is not None else False
    if framework == "vllm":
        llm = LLM(model=pretrained_model_path, dtype="bfloat16", max_num_seqs=batch_size * number_of_generation_per_sample, enable_lora=enable_lora, max_lora_rank=32)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
        if lora_adapter_path:
            model.load_adapter(lora_adapter_path)
        model.half().to(device)
        model.eval()
        llm = None

    exebench_evaluate(
        programs,
        prompter,
        tokenizer,
        model,
        result_file=result_file,
        batch_size=batch_size,
        llm=llm,
        number_of_generation_per_sample=number_of_generation_per_sample,
        lora_adapter_path=lora_adapter_path,
        remove_comments=remove_comments)


if __name__ == "__main__":
    fire.Fire(exebench_main)
