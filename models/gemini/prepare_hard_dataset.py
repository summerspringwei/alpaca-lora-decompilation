import json
import subprocess
import os
import tempfile
from datasets import load_from_disk, Dataset, DatasetDict

"""

Example of function info: {name: foo, 'unused_args': [], 'struct_args': False, 'has_globals': True, 'called_functions': ['mpt3sas_base_get_iocstate', 'scsi_host_busy', 'wait_event_timeout']}

"""
info_binary = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
# sample_dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"
dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"
saved_dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_hard"


exebench_data = load_from_disk(dataset_path)

bb_threshold = 3

def filter_func_info(record):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
        f.flush()
        cmd = [info_binary, f.name]
        cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
        if cmd_out.returncode != 0:
            print(f"Error Counting bb for: {record['llvm_ir']['code'][-1]} output: {cmd_out.stdout}")
            return False
        llvm_ir_info = cmd_out.stdout.decode("utf-8")
        out = json.loads(llvm_ir_info)
        filter_func = True
        bb_count = record['llvm_ir']['bb_count']["bbcount"]
        if bb_count > bb_threshold:
            return True
        # for func_dict in out['functions']:
        #     for _, func_info in func_dict.items():
        #         has_unused_args = True if len(func_info['unused_args']) > 0 else False
        #         has_struct_args = func_info['struct_args']
        #         has_call_functions = True if len(func_info['called_functions']) > 0 else False
                # if has_unused_args:
                #     filter_func = True
                # if has_call_functions:
                #     filter_func = False
                #     break


def filter_cannot_parse(record):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
        f.flush()
        cmd = [info_binary, f.name]
        cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
        if cmd_out.returncode != 0:
            return False
    return True


def map_func_info(record):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
        f.flush()
        cmd = [info_binary, f.name]
        cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
        llvm_ir_info = cmd_out.stdout.decode("utf-8")
        out = json.loads(llvm_ir_info)
        # record['llvm_ir']['func_info'] = out
        record['func_info'] = out
        return record




def filter_unused_args(record):
    for func_dict in record['func_info']['functions']:            
        has_unused_args = True if len(func_dict['unused_args']) > 0 else False
        if has_unused_args:
            return False
    return True


print("exebench_data", len(exebench_data))
# First map to add functions field
exebench_data = exebench_data.filter(filter_cannot_parse, num_proc=16)
print("exebench_data after filter_cannot_parse", len(exebench_data))
dataset_with_func_info = exebench_data.map(map_func_info, num_proc=16)
print("dataset_with_func_info", len(dataset_with_func_info))
dataset_filter_unused_args = dataset_with_func_info.filter(filter_unused_args, num_proc=16)
print("dataset_filter_unused_args", len(dataset_filter_unused_args))
# Then filter based on functions length
dataset_with_called_functions = dataset_filter_unused_args.filter(lambda record: len(record['func_info']["functions"][0]["called_functions"]) > 0, num_proc=16)
print("dataset_with_called_functions", len(dataset_with_called_functions))
bb_more_than_threshold = dataset_with_called_functions.filter(lambda record: record['llvm_ir']['bb_count']["bbcount"] > bb_threshold, num_proc=16)
print("bb_more_than_threshold", len(bb_more_than_threshold))
