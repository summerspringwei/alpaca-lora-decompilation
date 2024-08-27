import torch
from vllm import LLM, SamplingParams
if torch.cuda.is_available():
        device = "cuda"
else:
        device = "cpu"

path = "/home/xiachunwei/Datasets/Models/CodeLlama-7b-hf/"
# llm = LLM(model=path)
# PROMPT = '''def remove_non_ascii(s: str) -> str:
#         """ <FILL_ME>
#             return result
#             '''
# input_list = [PROMPT, PROMPT, PROMPT, PROMPT]
# sampling_params = SamplingParams(
#         n=16,
#         # top_k=-1,
#         # temperature=0,
#         # top_p=1,
#         max_tokens=128,
#         min_tokens=16,
#         # temperature=0.8, top_p=0.95
#         temperature=0, top_p=1,
#         use_beam_search=True
#         )
# sequences = llm.generate(input_list, sampling_params)
# for seq in sequences:
#     print("----"*20)
#     for out in seq.outputs:
#         print(out.text)
# # print(sequences)


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    path, num_labels=1, torch_dtype=torch.bfloat16
)
print(model)

# from transformers import AutoTokenizer
# import transformers
# import torch

# model = "facebook/llm-compiler-7b-ftd"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# sequences = pipeline(
#     '%3 = alloca i32, align 4',
#     do_sample=True,
#     top_k=10,
#     temperature=0.1,
#     top_p=0.95,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

