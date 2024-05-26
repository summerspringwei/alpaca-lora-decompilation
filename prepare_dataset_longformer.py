
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer
)

# dataset = load_dataset("json", data_files="/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat.json")

# dataset = dataset['train']

# dataset = dataset.train_test_split(test_size = 0.1)
# dataset_train = dataset['train']
# dataset_val = dataset['test']

# encoder_max_length = 8192
# decoder_max_length = 8192

# # load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")

# # max encoder length is 8192 for PubMed

# def process_data_to_model_inputs(row):
#     # tokenize the inputs and labels
#     inputs = tokenizer(
#         row["input"]
#     )
#     row["input_length"] = len(inputs.input_ids)
#     return row

# dataset_train = dataset_train.map(process_data_to_model_inputs, num_proc=40)
# dataset_val = dataset_val.map(process_data_to_model_inputs, num_proc=40)
# dataset_train = dataset_train.sort("input_length")
# dataset_val = dataset_val.sort("input_length")
# dataset_train.save_to_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train")
# dataset_val.save_to_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val")


dataset_train = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort").select(range(0, 1024))
dataset_val = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort").select(range(0, 1024))

encoder_max_length = 8192
decoder_max_length = 8192
batch_size = 10240
window_size = 1024

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
global_head_num_attn = 64
global_tail_num_attn = 256

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["input"],
        padding="longest",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["output"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    if global_head_num_attn < len(batch["input_ids"][0]) and len(batch["input_ids"][0]) > window_size:
        for i in range(len(batch["global_attention_mask"])):
            for j in range(0, global_head_num_attn):
                batch["global_attention_mask"][i][j] = 1
            for j in range(0, global_tail_num_attn):
                batch["global_attention_mask"][i][-1-j] = 1
    
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# map train data
dataset_train = dataset_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["input", "output", "instruction"],
    num_proc=40,
)

# map val data
dataset_val = dataset_val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["input", "output", "instruction"],
    num_proc=40,
)

# set Python list to PyTorch tensor
dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "file"],
)

# set Python list to PyTorch tensor
dataset_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "file"],
)

dataset_train.save_to_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort_tokenized_max_padding")
dataset_val.save_to_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort_tokenized_max_padding")
