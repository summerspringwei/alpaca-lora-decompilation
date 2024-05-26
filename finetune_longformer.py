#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Any, Union
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import torch
from torch import nn
import transformers
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase
)
from transformers.utils.generic import PaddingStrategy
from transformers.models.led.modeling_led import LEDLearnedPositionalEmbedding
# from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):
        encoder_max_length = 8192
        decoder_max_length = 8192
        global_head_num_attn = 64
        global_tail_num_attn = 256
        window_size = 1024
        input_list = [f["input"] for f in batch]
        inputs = tokenizer(
            input_list,
            padding="longest",
            truncation=True,
            max_length=encoder_max_length,
        )
        output_list = [f["output"] for f in batch]
        outputs = tokenizer(
            output_list,
            padding="longest",
            truncation=True,
            max_length=decoder_max_length,
        )
        features = {}
        features["input_ids"] = torch.tensor(inputs.input_ids, dtype=torch.long)
        features["attention_mask"] = torch.tensor(inputs.attention_mask, dtype=torch.long)

        # create 0 global_attention_mask lists
        features["global_attention_mask"] = len(features["input_ids"]) * [
            [0 for _ in range(len(features["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        if global_head_num_attn < len(features["input_ids"][0]) and len(features["input_ids"][0]) > window_size:
            for i in range(len(features["global_attention_mask"])):
                for j in range(0, global_head_num_attn):
                    features["global_attention_mask"][i][j] = 1
                for j in range(0, global_tail_num_attn):
                    features["global_attention_mask"][i][-1-j] = 1
        features["global_attention_mask"] = torch.tensor(features["global_attention_mask"], dtype=torch.long)
        features["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        features["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in features["labels"]
        ]
        features["labels"] = torch.tensor(features["labels"], dtype=torch.long)
        return features


# load rouge
rouge = load_metric("rouge")

model_path = "/data/xiachunwei/Datasets/Models/led-large-16384"

batch_size = 8
dataset_train = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort")
dataset_val = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort")
dataset_train.remove_columns("file")
dataset_val.remove_columns("file")

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    output_dir="./",
    logging_steps=250,
    eval_steps=5000,
    save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    remove_unused_columns=False
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# load model + enable gradient checkpointing & disable cache for checkpointing
led = AutoModelForSeq2SeqLM.from_pretrained(model_path, gradient_checkpointing=True, use_cache=False)

# Note, important, change the decoder module's position embedding size to 8192*1024
led.base_model.decoder.embed_positions = LEDLearnedPositionalEmbedding(8192, 1024)
nn.init.xavier_uniform_(led.base_model.decoder.embed_positions.weight)

# set generate hyperparameters
led.config.num_beams = 4
led.config.max_length = 8192
led.config.min_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3



# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=MyDataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

# start training
trainer.train()
