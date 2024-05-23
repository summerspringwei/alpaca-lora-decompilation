
import os
from typing import List

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_model(
    load_8bit: bool = False,
    base_model: str = ""
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
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
    
    return tokenizer, model




def evaluate_batch(
        tokenizer, model,
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
        # prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(instruction_list, return_tensors="pt", padding=True, truncation=True)
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
            
        return outputs


def main():
    tokenizer, model = get_model(base_model="/home/xiachunwei/Dataset/deepseek-coder-1.3b-instruct")
    outputs = evaluate_batch(
        tokenizer, model, [
         "Help me write a python function for quick sort: def quick_sort(arr: List[int]):",
         "Help me write a python function for hello world",
        ]
    )
    print(outputs)
    
if __name__ == "__main__":
    main()
    