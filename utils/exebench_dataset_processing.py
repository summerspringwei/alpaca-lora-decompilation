import os
import json
import subprocess
import tempfile
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict

"""
Example of function info: {'name': 'foo', 'unused_args': [], 
    'struct_args': False, 'has_globals': True, 
    'called_functions': ['mpt3sas_base_get_iocstate', 'scsi_host_busy', 'wait_event_timeout']}
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    # Try to find llvm-parser-checker in PATH
    which_cmd = subprocess.run(['which', 'llvm-parser-checker'], 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    if which_cmd.returncode == 0:
        info_binary = which_cmd.stdout.decode('utf-8').strip()
    else:
        info_binary = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
except:
    # Keep default value if which command fails
    pass

tokenizer = AutoTokenizer.from_pretrained("/home/xiachunwei/Dataset/Qwen3-0.6B", trust_remote_code=True)

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
        record['func_info'] = out
        tokenized_question = tokenizer(record['asm']['code'][-1])
        record['token_length'] = len(tokenized_question["input_ids"])
    
        return record


def filter_unused_args(record):
    for func_dict in record['func_info']['functions']:
        has_unused_args = True if len(func_dict['unused_args']) > 0 else False
        if has_unused_args:
            return False
    return True


def filter_with_struct_args(record):
    for func_dict in record['func_info']['functions']:
        if func_dict['has_defined_structs']:
            return True
    return False


def dump_llvm_ir(exebench_data, saved_dataset_path):
    for idx, record in enumerate(exebench_data):
        with open(os.path.join(saved_dataset_path, f"{idx}.ll"), "w") as f:
            f.write(f";cpath: {record['path']}\n" + record['llvm_ir']['code'][-1])


def filter_num_of_instructions(record, threshold=50):
    filter_num_of_instructions = sum(record['llvm_ir']['bb_count']['bb_list_size'])
    return True if filter_num_of_instructions > threshold else False


def filter_has_loops(record):
    for func_dict in record['func_info']['functions']:
        if func_dict['num_loops'] > 0:
            return True
    return False
