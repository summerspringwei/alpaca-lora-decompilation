import json

def load_and_print(file_path):
    data = json.load(open(file_path, 'r'))
    for record in data:
        print("\n")
        print("/*", record["file:"], "*/")
        print(record["output"])
        print("\n")
        print("//"*20)
        print(record["predict"])
        print("\n")
        print("/", "*"*20, "/")

# load_and_print("result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K_our_lora.json")

# load_and_print("result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample128.json")

load_and_print("result_AnghaBench_instruct_train_paired_assembly-g-O2_C_2K_sample_tail1K_codellama.json")
