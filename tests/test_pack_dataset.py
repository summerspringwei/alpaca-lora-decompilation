import random
import torch
from typing import Union

from datasets import load_from_disk
from datasets import Dataset
from transformers import AutoTokenizer


from utils.prompter import Prompter
from utils.preprocessing_assembly import preprocessing_assembly

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def prepare_dataloader(dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
    """
    Prepare the dataloader for training.

    Args:
        dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
            PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
            will be preprocessed by removing the columns that are not used by the model.
        data_collator (Optional[function]):
            Data collator function.

    Returns:
        `torch.utils.data.DataLoader`: PyTorch dataloader
    """
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=True,
    )
    return dataloader


path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_1_llvm_extract_func_ir_assembly_O2_llvm_diff"
train_dataset = dataset = load_from_disk(path_to_dataset)
tokenizer = AutoTokenizer.from_pretrained("/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd")


# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
input_max_length=512
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

instruction_str = "Disassemble this code to LLVM-IR"
prompter = Prompter("llmcompiler")

def pack_similar_length_samples(dataset, batch_size: int, num_chunks: int):
    """
    Pack similar length samples into the same batch. This is to reduce the padding size.

    Args:
        dataset (`torch.utils.data.Dataset`):
            The dataset to be packed.

        batch_size (`int`):
            The batch size.

        num_chunks (`int`):
            The number of chunks to be packed.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the packed dataset.
    """
    if len(dataset) <= batch_size * num_chunks * 2:
        return dataset
    sorted_dataset = sorted(dataset, key=lambda x: len(x["input_ids"]))
    batched_dataset = [sorted_dataset[i: i + batch_size] for i in range(0, len(sorted_dataset), batch_size)]
    # shape: (num_chunks, batch_size)
    # Split to num_chunks
    chunks_size = len(batched_dataset) // num_chunks
    chunks_chunks = [batched_dataset[i: i+chunks_size] for i in range(0, len(batched_dataset), chunks_size)]
    # Shape: (num_chunks, chunks_size, batch_size)
    
    # Extract respectively from chunks
    interleved_batch = []
    for j in range(chunks_size):
        for i in range(num_chunks):
            interleved_batch.append(chunks_chunks[i][j])
    random.shuffle(interleved_batch)
    new_sequences = []
    for b in interleved_batch:
        new_sequences.extend(b)
    new_dataset = Dataset.from_list(new_sequences)
    new_dataset.set_format(type="torch")
    return new_dataset


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
    ds = ds.filter(
        lambda x: len(x["input_ids"]) < input_max_length,
        batched=False,
        num_proc=num_proc)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)

dataset = pack_similar_length_samples(dataset, 16, 8)

# Save dataset that is large than 500 to disk
path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_1_llvm_extract_func_ir_assembly_O2_llvm_diff_input_len_500"

dataset = dataset.filter(
    lambda x: len(x["input_ids"]) > 500,
    batched=False,
    num_proc=16).select(range(32))
print(len(dataset))
dataset.save_to_disk(path_to_dataset)
# for i, batch in enumerate(dataset):
#     if i % 8 == 0:
#         print(batch["input_ids"].shape)

# print("aaa")
# datacollator = prepare_dataloader(dataset, data_collator=collator)
# for epoch, batch in enumerate(datacollator):
#     for b in batch["input_ids"]:
#         print(b.shape)
    