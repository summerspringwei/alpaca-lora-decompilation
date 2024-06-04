
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer
)

dir_path = "/data/xiachunwei/Datasets/decompilation-dataset/"
dataset_path = f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat.json"
train_dataset_path = f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train"
val_dataset_path = f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val"


encoder_max_length = 8192
decoder_max_length = 8192
batch_size = 10240
window_size = 1024

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
global_head_num_attn = 64
global_tail_num_attn = 256

def sort_dataset_by_length_and_split(dataset_path, train_dataset_path, val_dataset_path):
    dataset = load_dataset("json", data_files=dataset_path)

    dataset = dataset['train']

    dataset = dataset.train_test_split(test_size = 0.1)
    dataset_train = dataset['train']
    dataset_val = dataset['test']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")

    # max encoder length is 8192 for PubMed

    def process_data_to_model_inputs(row):
        # tokenize the inputs and labels
        inputs = tokenizer(
            row["input"]
        )
        row["input_length"] = len(inputs.input_ids)
        return row

    dataset_train = dataset_train.map(process_data_to_model_inputs, num_proc=40)
    dataset_val = dataset_val.map(process_data_to_model_inputs, num_proc=40)
    dataset_train = dataset_train.sort("input_length")
    dataset_val = dataset_val.sort("input_length")
    dataset_train.save_to_disk(train_dataset_path)
    dataset_val.save_to_disk(val_dataset_path)


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


def format_dataset(in_dataset_path, out_dataset_path):
    dataset = load_from_disk(in_dataset_path)

    # map train data
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "output"],
        num_proc=40,
    )

    # set Python list to PyTorch tensor
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    dataset.save_to_disk(out_dataset_path)


def format_train_val():
    format_dataset(f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort_tokenized", 
                   f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort_tokenized_max_padding")
    format_dataset(f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort_tokenized",
                     f"{dir_path}/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort_tokenized_max_padding")


def main():
    sort_dataset_by_length_and_split(dataset_path, train_dataset_path, val_dataset_path)
    format_train_val()


if __name__ == "__main__":
    main()
