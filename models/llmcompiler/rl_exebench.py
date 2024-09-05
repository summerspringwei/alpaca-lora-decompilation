# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
import json
from dataclasses import dataclass, field
from typing import Optional


import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from utils.prompter import Prompter
from utils.evaluate_exebench import validate_by_execution
from utils.preprocessing_assembly import preprocessing_assembly
tqdm.pandas()


def dump_generation_results(batch, f_generate):
    for path, query, response, ir, func_head_types in zip(batch["path"], batch["query"], batch["response"], batch["llvm_ir"], batch["func_head_types"]):
        json_output = {
            "instruction": "Disassemble this code to LLVM-IR",
            "input": query,
            "predict": response,
            "file": path,
            "output": ir["code"][0],
            "func_head_types": func_head_types
        }
        json.dump(json_output, f_generate, indent=4,
                              sort_keys=True,
                              separators=(',', ':'))
        f_generate.write(",\n")
    f_generate.flush()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    output_max_length: Optional[int] = field(default=1548, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=2, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=100, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=3200, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 8bit"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name


config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    remove_unused_columns=False, # We need to keep the original columns for the reward model
)

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

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
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
            assembly_str = preprocessing_assembly(record["code"][-1])
            query = prompter.generate_prompt(instruction_str, assembly_str)
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
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.input_max_length, batched=False, num_proc=num_proc)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 16,
    "temperature": 0.1,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 2, # tokenizer.eos_token_id only valid for llama2
    "max_new_tokens": script_args.output_max_length,
    "batch_size": script_args.batch_size,
}
# output_min_length = 32
# output_max_length = script_args.output_max_length
output_length_sampler = None
# output_length_sampler = LengthSampler(output_min_length, output_max_length)


result_file = f"ppo-llm-compiler-bs-{script_args.batch_size}-mbs-{script_args.mini_batch_size}.json"
f_generate = open(result_file + "_inc", "a")

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break
    question_tensors = batch["input_ids"]
    
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Create dummy response for debugging
    # batch["response"] = ["a" for _ in batch["query"]]
    dump_generation_results(batch, f_generate)
    # For debug only, we dump the intermedia results, so that we can avoid the time consuming generating process
    # pickle.dump(question_tensors, open("question_tensors.pkl", 'wb'))
    # pickle.dump(response_tensors, open("response_tensors.pkl", 'wb'))
    # pickle.dump(batch, open("texts.pkl", 'wb'))
    # batch = pickle.load(open("texts.pkl", 'rb'))
    # response_tensors = pickle.load(open("response_tensors.pkl", 'rb'))
    # question_tensors = pickle.load(open("question_tensors.pkl", 'rb'))
    
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    
    # Compute reward score (using the sentiment analysis pipeline)
    # Get the reward from compiler and execution
    predict_results = [{key: value[idx] for key, value in batch.items()} for idx in range(len(batch['path']))]
    records = [{'predict':row['response'], 'file': row['path'], 'output': row['llvm_ir']['code'][-1]} for row in predict_results]
    validation_results = [validate_by_execution(record, row) for row, record in zip(predict_results, records)]
    rewards = []
    for output in validation_results:
        reward = -1
        if not output["target_execution_success"]:
            reward = 0
        elif output["predict_execution_success"]:
            reward = 1
        elif output["predict_compile_success"]:
            reward = -0.1
        else:
            reward = -1
        rewards.append(reward)
    rewards = [torch.tensor(output, dtype=torch.float32) for output in rewards]
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
