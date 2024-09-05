
import pickle
import json
from dataclasses import dataclass, field
from typing import Optional


import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
import transformers
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
print(transformers.__file__)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from utils.prompter import Prompter
from utils.evaluate_exebench import validate_by_execution
tqdm.pandas()
input_max_length = 384
model_dir = "/home/xiachunwei/Datasets/Models/CodeLlama-7b-hf/"
path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_1_llvm_extract_func_ir_assembly_O2"
train_dataset = dataset = load_from_disk(
    path_to_dataset
)
# train_dataset = train_dataset.select(range(1000))
original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 8,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(model_dir)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

instruction_str = "Disassemble this code to LLVM-IR"
prompter = Prompter("llmcompiler")

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # num_proc = 24
    num_proc = 1

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for record in examples["asm"]:
            query = prompter.generate_prompt(instruction_str, record["code"][-1])
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        # remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < input_max_length, batched=False, num_proc=num_proc)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)
# dataset = dataset.select(range(64))
ratio_list = []
for record in dataset:
    ir_len = len(tokenizer(record["llvm_ir"]["code"][0])["input_ids"])
    ratio = ir_len / len(record["input_ids"])
    ratio_list.append(ratio)
    # print(len(record["input_ids"]), ir_len, ratio)
import numpy as np
print(np.max(ratio_list), np.min(ratio_list), np.mean(ratio_list), len(ratio_list))