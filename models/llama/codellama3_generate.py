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
