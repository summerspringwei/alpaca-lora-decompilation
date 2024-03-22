import torch
print(torch.__version__)

# import mii
# pipe = mii.pipeline("/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora")
# response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
# print(response)

from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")