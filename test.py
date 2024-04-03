import torch
print(torch.__version__)
import transformers
print(transformers.__version__)

print(2560*128)
print(2560*128 + 40960)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
base_model="/data0/xiachunwei/Dataset/deepseek-coder-1.3b-base-angha-llvm-ir"
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        ).cuda()

tokenizer = AutoTokenizer.from_pretrained(base_model)
print(tokenizer.decode([6707]))
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# import mii
# pipe = mii.pipeline("/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora")
# response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
# print(response)

# from vllm import LLM, SamplingParams
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora")

# outputs = llm.generate(prompts, sampling_params)

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")