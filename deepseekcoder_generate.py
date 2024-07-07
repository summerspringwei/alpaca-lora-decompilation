
import os
import json
from typing import List, Dict

import fire
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils.prompter import Prompter
from datasets import load_dataset, load_from_disk
from safetensors import safe_open

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_deepseekcoder_model(pretrained_model_path: str, finetuned_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    model.config.pad_token_id = tokenizer.pad_token_id = 32014  # unk
    model.config.bos_token_id = 32013
    model.config.eos_token_id = 32014
    state_dict = {}
    with safe_open(finetuned_model_path, framework="pt", device="cuda:0") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    model = model.to(torch.bfloat16)

    return model, tokenizer



def get_model(
    load_8bit: bool = False,
    base_model: str = "",
    prompt_template: str = None,  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = 'left'
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 32014  # unk
    model.config.bos_token_id = 32013
    model.config.eos_token_id = 32014

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    
    return prompter, tokenizer, model


def evaluate(
        prompter, tokenizer, model,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=50,
        num_beams=4,
        max_new_tokens=2048+1024,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            max_length=4096+1024,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.config.bos_token_id,
            **kwargs,
        )
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=False
            )
        if num_beams > 1:
            result = [prompter.get_response(tokenizer.decode(s)) for s in generation_output.sequences]
        else:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            result = prompter.get_response(output)
        return result



def evaluate_batch(
        prompter, tokenizer, model,
        instruction_list: List[str],
        input_list: List[str]=None,
        temperature=0.1,
        top_p=0.75,
        top_k=50,
        num_beams=1,
        max_new_tokens=2048+1024,
        **kwargs,
    ):
        """Currently we only support generate one candidate for each input."""
        prompt_list = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instruction_list, input_list)]
        # prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            max_length=4096+1024,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.config.bos_token_id,
            **kwargs,
        )
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=False
            )
            s = generation_output.sequences
            outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
            results = [prompter.get_response(o) for o in outputs]
        return results


def main(load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    val_file: str = "val.json",
    result_file: str = "val_result.json",
    start_idx: int = 0,
    end_idx: int = 0,
    batch_size: int = 1
    ):
    # programs = json.load(open(val_file))
    programs = load_dataset("json", data_files=val_file).shuffle()["train"]
    print(len(programs))
    if end_idx == 0:
        end_idx = len(programs)
    programs = programs.select(range(start_idx, end_idx))

    prompter, tokenizer, model = get_model(
        load_8bit, base_model, lora_weights, prompt_template
    )

    results = []
    count = len(programs) // batch_size + 1
    with open(result_file+"_inc", "a") as f:
        for i in range(count):
            try:
                start = i * batch_size
                end = min((i+1) * batch_size, len(programs))
                batch_p = [p["instruction"] for p in programs.select(range(start, end))]
                batch_input = [p["input"] for p in programs.select(range(start, end))]
                batch_predict = evaluate_batch(prompter, tokenizer, model, batch_p, input_list=batch_input)
                batch_output = [
                    {"instruction": p["instruction"], "input":p["input"], "predict": predict, "file": p["file"], "output": p["output"]}
                    for p, predict in zip(programs.select(range(start, end)), batch_predict)
                ]
                results.extend(batch_output)
                for p in batch_output:
                    json.dump(p, f, indent=4, sort_keys=True, separators=(',', ':'))
                    f.write(",\n")
            except Exception as e:
                print(e)
                continue
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True, separators=(',', ':'))


def exebench_evaluate(
    programs,
    prompter, tokenizer, model,
    result_file: str = "val_result.json",
    batch_size: int = 1
    ):
    results = []
    count = len(programs) // batch_size + 1
    with open(result_file+"_inc", "a") as f:
        for i in range(count):
            try:
                start = i * batch_size
                end = min((i+1) * batch_size, len(programs))
                batch_p = ["decompile the x86 assembly to llvm ir"] * batch_size
                batch_input = [p["asm"]["code"][-1] for p in programs.select(range(start, end))]
                batch_predict = evaluate_batch(prompter, tokenizer, model, batch_p, input_list=batch_input)
                batch_output = [
                    {"instruction":"decompile the x86 assembly to llvm ir", 
                     "input":p["asm"]["target"][-1], "predict": predict, "file": p["path"], "output": p["llvm_ir"]["code"]}
                    for p, predict in zip(programs.select(range(start, end)), batch_predict)
                ]

                results.extend(batch_output)
                for p in batch_output:
                    json.dump(p, f, indent=4, sort_keys=True, separators=(',', ':'))
                    f.write(",\n")
                    f.flush()
            except Exception as e:
                print(e)
                continue
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True, separators=(',', ':'))



def exebench_main():
    pretrained_model_path = "/data/xiachunwei/Datasets/Models/deepseek-coder-1.3b-base"
    finetuned_model_path = "/data/xiachunwei/Projects/checkpoints-decompilation/decompile_llvm_ir_alphaca_deepseek-coder-1.3b-base_seq_len_4k-bbcount-2/checkpoint-2900/model.safetensors"
    model, tokenizer = get_deepseekcoder_model(pretrained_model_path, finetuned_model_path)
    prompter = Prompter()
    path = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_ir_assembly_O2/"
    programs = load_from_disk(path)
    exebench_evaluate(programs, prompter, tokenizer, model, result_file="exebench_train_synth_rich_io_filtered_llvm_ir", batch_size=8)



if __name__ == "__main__":
    # fire.Fire(main)
    exebench_main()
