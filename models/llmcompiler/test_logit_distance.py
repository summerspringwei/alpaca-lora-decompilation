import json
from transformers import LlamaTokenizer, LlamaForCausalLM
from reward_functions import get_logits_distance
import torch

validation_file_path = "/home/xiachunwei/Projects/alpaca-lora-decompilation/validation/exebench_llmcompiler/exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-length_reward_pack_similar_length_samples-step-40-bs-32-beams-1_validate_exebench.json"
model_path = "/home/xiachunwei/Datasets/Models/llm-compiler-13b-ftd"
records = json.load(open(validation_file_path, 'r'))
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"  # Padding on the left side
tokenizer.pad_token = tokenizer.eos_token  # Use the eos token as the padding token
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") # Load in float16 for memory efficiency


start, batch = 0, 1
for start in range(0, len(records), batch):
    group = get_logits_distance(records[start:start+batch], tokenizer, model)
    for g in group:
        print(g["distance"], g["predict_compile_success"], g["predict_execution_success"])
        if not g["predict_compile_success"][0]:
            print(g["predict"][0])
            print("==="*20)
            print(g["output"])
    exit(0)