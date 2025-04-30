"""
This script is used to test the function information extraction from LLVM IR files.
It uses the llvm-parser-checker binary to extract function information and
count the number of basic blocks and instructions in the LLVM IR code.
"""
import subprocess
import json
from datasets import load_from_disk

info_binary = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
samples = load_from_disk("/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100")

for idx, record in enumerate(samples):
    file_name = f"dir_tmp/target{idx}.ll"
    with open(file_name, 'w') as f:
        f.write(record['llvm_ir']['code'][-1])
    cmd = [info_binary, file_name]
    cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
    if cmd_out.returncode != 0:
        print(f"Error Counting bb for: {record['llvm_ir']['code'][-1]} output: {cmd_out.stdout}")
        continue
    llvm_ir_info = cmd_out.stdout.decode("utf-8")
    out = json.loads(llvm_ir_info)
    info = out["functions"][0]

    # print number of BB and instructions
    bb_list = record['llvm_ir']["bb_count"]['bb_list_size']
    bb_count = record['llvm_ir']['bb_count']["bbcount"]
    num_instructions = sum(bb_list)
    # print(f"Num BB: {bb_count}, num instr: {num_instructions}")
    
    for name, value in info.items():
        if len(value["unused_args"]) > 0:
            print(llvm_ir_info)
            print(value["unused_args"])
        print(value["called_functions"])
        if len(value["called_functions"]) > 0:
            print(value["called_functions"])

import matplotlib.pyplot as plt

# Collect data for plotting
bb_counts = []
instruction_counts = []

for record in samples:
    bb_list = record['llvm_ir']["bb_count"]['bb_list_size']
    bb_count = record['llvm_ir']['bb_count']["bbcount"]
    num_instructions = sum(bb_list)
    bb_counts.append(bb_count)
    instruction_counts.append(num_instructions)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(bb_counts, instruction_counts, alpha=0.6, edgecolors='k')
plt.title("Distribution of Number of BB and Number of Instructions")
plt.xlabel("Number of Basic Blocks (BB)")
plt.ylabel("Number of Instructions")
plt.grid(True)
# plt.show()
plt.savefig("bb_vs_inst.png")
plt.close()