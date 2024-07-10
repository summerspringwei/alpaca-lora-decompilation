from datasets import load_from_disk
import json
a = json.load(open("exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json",'r'))
print(len(a))
exit(0)
path = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"
programs = load_from_disk(path)

# for p in programs.select(range(0, 1000)):
for p in programs:
    # print(len(p["llvm_ir"]["code"]))
    print(p['path'])
    # print(p["asm"]["code"][-1])
    print(len(p["asm"]["target"]), len(p["asm"]["code"]))
    print(p["asm"]["target"][-1])
    print(p["asm"]["code"][-1])
    print("="*20)
    
