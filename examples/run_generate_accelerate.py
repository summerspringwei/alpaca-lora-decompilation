import os
import sys
import json


import fire
import torch
import transformers
import accelerate
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader


# asserttorch.cuda.is_available():
#     device = "cuda"
accelerator = Accelerator()
device = accelerator.device

accelerator = Accelerator()

def get_model(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = None,  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        # device_map="auto",
    )
    if lora_weights is not None and lora_weights != "":
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            # device_map="auto",
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

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
        top_k=40,
        num_beams=8,
        max_new_tokens=3096,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_length=4096+128,
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
                max_new_tokens=max_new_tokens,
                early_stopping=True
            )
        if num_beams > 1:
            result = [prompter.get_response(tokenizer.decode(s)) for s in generation_output.sequences]
        else:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            result = prompter.get_response(output)
        return result


def main(load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    val_file: str = "val.json",
    result_file: str = "val_result.json",
    start_idx: int = 0,
    end_idx: int = 0,
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
    model, eval_dataloader = accelerator.prepare(model, DataLoader(programs, batch_size=1))
    import pdb
    pdb.set_trace()
    results = []
    with open(result_file+"_inc", "w") as f:
        for p in tqdm(eval_dataloader):
            predict = evaluate(prompter, tokenizer, model, p["instruction"], input=p["input"])
            val_out = {"instruction": p["instruction"], "input":p["input"], "predict": predict, "file": p["file"], "output": p["output"]}
            results.append(val_out)
            print("="*20)
            print(p["output"])
            print("-"*20)
            print(predict)
            print("="*20)
            json.dump(val_out, f, indent=4, sort_keys=True, separators=(',', ':'))
            f.write(",\n")
    with open(result_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(main)
