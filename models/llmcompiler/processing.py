
import random
from typing import Union

import torch
import transformers
from datasets import Dataset

from utils.prompter import Prompter
from utils.preprocessing_assembly import preprocessing_assembly

instruction_str = "Disassemble this code to LLVM-IR"
prompter = Prompter("llmcompiler")

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    train_dataset: Union[torch.utils.data.Dataset, Dataset],
    input_max_length: int = 384
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
            # assembly_str = assembly_str * 4 # Debug only
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


def pack_similar_length_samples(dataset, batch_size: int, num_chunks: int, sort_key_function=lambda x: len(x["input_ids"])):
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
    sorted_dataset = sorted(dataset, key=sort_key_function)
    batched_dataset = [sorted_dataset[i: i + batch_size] for i in range(0, len(sorted_dataset), batch_size)] # Shape: (num_batches, batch_size)
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
