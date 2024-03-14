# # import torch
# import json
# # # a = torch.load("xcw_alphaca_lora/adapter_model.bin")
# # # print(a)


# # # print(sum([1, 2]))
# # mlist = [1,2]
# # bound = (lambda self, *_, **__: print(sum(self))).__get__(mlist, type(mlist))
# # print(bound())

# def read_file(file_path):
#     programs = json.load(open(file_path))
#     for p in programs:
#         print(p["instruction"])
        

# if __name__=="__main__":
#     read_file("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample128.json")


# import json
# import time

# # Sample data
# data = [
#     {"name": "Alice", "age": 30},
#     {"name": "Bob", "age": 25},
#     {"name": "Charlie", "age": 35}
# ]

# # File path to save JSON data
# file_path = "aaaaadata.json"

# # Open the file in write mode
# with open(file_path, "w") as json_file:
#     # Iterate over each object in the data list
#     for obj in data:
#         # Write the JSON object to the file
#         json.dump(obj, json_file)
#         # Add a newline character to separate objects
#         json_file.write('\n')
#         time.sleep(5)



from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
    load_peft_weights
)

import torch
checkpoint_name = "/data0/xiachunwei/Projects/alpaca-lora/decompile_llvm_ir_alphaca_lora/checkpoint-6500/adapter_model.safetensors"
# torch.load(checkpoint_name)
# out = load_peft_weights(checkpoint_name)
from safetensors import safe_open
f = safe_open(checkpoint_name, framework="pt", device="cpu")
tensors = {}
for k in f.keys():
    print(k)
    tensors[k] = f.get_tensor(k)

print(tensors)
