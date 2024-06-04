import torch
from transformers import LongformerModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/xiachunwei/Dataset/longformer-base-4096")
SAMPLE_TEXT = " ".join(["Tell me a story about a singer a" * 255])  # long input document
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
model = LongformerModel.from_pretrained("/home/xiachunwei/Dataset/longformer-base-4096")


attention_mask = torch.ones(
    input_ids.shape, dtype=torch.long, device=input_ids.device
)  # initialize to local attention
global_attention_mask = torch.zeros(
    input_ids.shape, dtype=torch.long, device=input_ids.device
)  # initialize to global attention to be deactivated for all tokens

global_attention_mask[
            :,
            [
                1,
                6,
                7,
                8
            ],
        ] = 1

outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
print(outputs)
print(outputs.last_hidden_state.shape)
print(outputs.global_attention_mask.shape)
print(outputs.attentions.shape)
print(outputs.pooler_output.shape)
