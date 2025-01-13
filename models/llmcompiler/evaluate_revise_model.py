
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader

from models.llmcompiler.llmcompiler_generate import vllm_evaluate_batch
from utils.prompter import Prompter

path_to_dataset = "/home/xiachunwei/Datasets/revise_exebench/revised_exebench_split_03456"
lora_adapter_path = "/home/xiachunwei/Projects/alpaca-lora-decompilation/lora_revise_exebench_split_03456/checkpoint-102"
pretrained_model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-7b-ftd"

dataset = load_from_disk(path_to_dataset)
prompter = Prompter("alpaca")
batch_size = 8
enable_lora = True
llm = LLM(model=pretrained_model_path, max_num_seqs=batch_size, enable_lora=enable_lora)


dataloader = DataLoader(
    dataset["train"],
    batch_size=batch_size,
    shuffle=False
)

for datapoint in dataloader:
    instruction_list = datapoint["instruction"]
    input_list = datapoint["input"]
    results = vllm_evaluate_batch(
        prompter,
        llm,
        instruction_list,
        input_list,
        temperature=0.1,
        top_p=0.95,
        top_k=10,
        num_beams=1,
        max_new_tokens=1024 * 4,
        lora_adapter_path=lora_adapter_path,
    )
    print(results)
