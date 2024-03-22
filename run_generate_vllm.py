import os
import json


import fire
import torch
import transformers

from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from datasets import load_dataset
from vllm import LLM, SamplingParams


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_model(
    base_model: str = "",
    prompt_template: str = None,  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    model = LLM(model=base_model)

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    # model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    
    return prompter, model


def evaluate(
        prompter,
        model,
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
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens = 2048)
        outputs = model.generate(prompt, sampling_params)
        outputs_text = [f"{out.prompt} {out.outputs[0].text}" for out in outputs]
        return outputs_text
        


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

    prompter, model = get_model(
        base_model, prompt_template
    )

    results = []
    
    with open(result_file+"_inc", "w") as f:
        for p in tqdm(programs):
            predict = evaluate(prompter, model, p["instruction"], input=p["input"])
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
