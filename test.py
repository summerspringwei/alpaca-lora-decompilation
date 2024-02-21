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


import json
import time

# Sample data
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# File path to save JSON data
file_path = "aaaaadata.json"

# Open the file in write mode
with open(file_path, "w") as json_file:
    # Iterate over each object in the data list
    for obj in data:
        # Write the JSON object to the file
        json.dump(obj, json_file)
        # Add a newline character to separate objects
        json_file.write('\n')
        time.sleep(5)