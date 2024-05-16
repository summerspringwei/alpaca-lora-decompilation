import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(
    "/data/xiachunwei/Datasets/Models/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "/data/xiachunwei/Datasets/Models/Meta-Llama-3-8B-Instruct")
tokenizer.add_special_tokens({"pad_token":"<pad>"})
# generation_config = GenerationConfig(
#     temperature=0.1,
#     top_p=0.76,
#     top_k=50,
#     do_sample=True,
#     num_return_sequences=1,
#     max_length=1024,
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=model.config.bos_token_id,
# )

pipeline = transformers.pipeline("text-generation",
                                 model=model,
                                 tokenizer=tokenizer,
                                 max_new_tokens=1024,
                                #  config=generation_config,
                                 model_kwargs={"torch_dtype": torch.bfloat16},
                                 device="cuda:1")
# Enter input from keyborad end with \n
# output = pipeline("Write a python function for quick sort")
# print(output[0]["generated_text"])
# exit(0)
while True:
    prompt = input("please enter prompt:").strip()
    if prompt == "exit" or prompt == "q":
        exit(0)
    output = pipeline(prompt)
    print(output[0]["generated_text"])

# import torch
# import torch.nn.functional as F

# class GroupedQueryAttention(torch.nn.Module):
#     def __init__(self, input_dim, num_clusters, num_heads):
#         super(GroupedQueryAttention, self).__init__()
#         self.input_dim = input_dim
#         self.num_clusters = num_clusters
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads

#         # Define linear layers for query, key, and value projections
#         self.query_linear = torch.nn.Linear(input_dim, input_dim)
#         self.key_linear = torch.nn.Linear(input_dim, input_dim)
#         self.value_linear = torch.nn.Linear(input_dim, input_dim)

#         # Define linear layer for output projection
#         self.output_linear = torch.nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()

#         # Project inputs to query, key, and value
#         query = self.query_linear(x)
#         key = self.key_linear(x)
#         value = self.value_linear(x)

#         # Split into multiple heads
#         query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # Group tokens into clusters
#         cluster_size = seq_len // self.num_clusters
#         query = query.view(batch_size, self.num_heads, self.num_clusters, cluster_size, self.head_dim).sum(dim=3)
#         key = key.view(batch_size, self.num_heads, self.num_clusters, cluster_size, self.head_dim).sum(dim=3)
#         value = value.view(batch_size, self.num_heads, self.num_clusters, cluster_size, self.head_dim).sum(dim=3)

#         # Compute attention scores
#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

#         # Apply softmax to compute attention weights
#         attention_weights = F.softmax(attention_scores, dim=-1)

#         # Apply attention weights to value
#         attention_output = torch.matmul(attention_weights, value)

#         # Merge heads and clusters
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

#         # Project attention output
#         output = self.output_linear(attention_output)

#         return output
