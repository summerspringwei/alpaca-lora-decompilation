from datasets import load_dataset, load_metric, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import torch
a = torch.ones([128, 1052, 64], dtype=torch.bfloat16)
b = torch.ones([128, 64, 1052], dtype=torch.bfloat16)
c = torch.bmm(a, b)
print(c.shape)
