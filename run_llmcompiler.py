from transformers import AutoTokenizer
import transformers
import torch
from vllm import LLM, SamplingParams

model = "/data/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"
llm = LLM(model=model)
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     device=0
# )

def get_multiline_input():
    print("Enter your input (type 'END' to finish):")
    lines = ["[INST] Disassemble this code to LLVM-IR: \n <code> ",]
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines) + " \n </code> [/INST]"


while True:
    message = get_multiline_input()
    sampling_params = SamplingParams(
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        max_tokens=4096,
        min_tokens=128)
    sequences = llm.generate([message,], sampling_params)
    # sequences = pipeline(
    #     message,
    #     do_sample=True,
    #     top_k=10,
    #     temperature=0.1,
    #     top_p=0.95,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=8192,
    # )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
