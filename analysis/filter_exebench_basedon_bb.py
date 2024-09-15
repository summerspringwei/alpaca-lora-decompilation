"""This module is for filter the dataset by the llvm-diff tool"""
import tqdm
import subprocess
import tempfile
import numpy as np

import fire
from datasets import load_from_disk, Dataset
from typing import Dict, List

from analysis.count_bb import set_bb_count, get_bulk_list


def write_to_tmp_file(code: str):
    f = tempfile.NamedTemporaryFile(delete=True, mode='w')
    f.write(code)
    f.flush()
    # f.close()
    return f


llvm_diff = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-mdiff"


def analyze_the_diff(diff: str)->bool:
    """Analyze the log of llvm-diff to determine whether the two llvm irs are different
    Args:
        diff: log of llvm-diff
    
    Returns:
        bool: whether the two llvm irs are different
    """
    diff_msg_list = ["different argument counts", 
                     "called functions differ",
                     "argument counts differ",
                     "different instruction types",
                     "different predicates",
                     "different phi types",
                     "PHI node # of incoming values differ",
                     "different number of incoming edges",
                     "PHI node incoming values differ",
                     "callbr # of indirect destinations differ",
                     "branch conditionality differs",
                     "branch conditions differ"
                     ]
    diff_msg_list = ["different", "differ"]
    diff_lines = diff.split("\n")
    diff = False
    for line in diff_lines:
        if line == "":
            continue
        if line.find("exists only in left module")>=0 or line.find("exists only in right module") >= 0:
            continue
        for msg in diff_msg_list:
            if line.find(msg) != -1:
                diff = True
                break
    return diff


def get_record_list_from_bb_count(record_list: List[Dict]) -> List[List[Dict]]:
    file_list = [write_to_tmp_file(record['llvm_ir']['code'][-1]) for record in record_list]
    same_type_list = [(record_list[0], file_list[0]), ]
    type_list = [same_type_list, ]
    # Iterate over all records
    for record, ir_file in tqdm.tqdm(zip(record_list[1:], file_list[1:]), total=len(record_list[1:])):
        find_same_type = False
        # Try to find same type in the current type list
        for current_type_list in type_list:
            f1 = current_type_list[0][1]
            f2 = ir_file
            cmd_out = subprocess.run([llvm_diff, f1.name, f2.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if cmd_out.returncode == 0:
                current_type_list.append((record, f2))
                find_same_type = True
                break
            elif cmd_out.returncode == 1:
                diff = cmd_out.stderr.decode("utf-8")
                print(diff)
            else:
                print(f"error diff {f1.name} and {f2.name}")
            # is_diff = analyze_the_diff(diff)
            # if not is_diff:
            #     current_type_list.append((record, f2))
            #     find_same_type = True
            #     break
        if not find_same_type:
            type_list.append([(record, f2), ])
    print(f"total {len(record_list)} alltypes {len(type_list)}")
    type_list.sort(key=lambda x: len(x), reverse=True)
    for same_type in type_list:
        # print(";"*20)
        print(";",len(same_type))
        # print(type[0][0]['llvm_ir']['code'][-1])
    return type_list


def filter_by_llvm_diff(path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2",
    result_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"):
    # 1. Load dataset and set the number of BB, 
    train_dataset = load_from_disk(
        path_to_dataset
    )
    train_dataset = train_dataset.map(set_bb_count, num_proc=8)
    bulk_len_record = get_bulk_list(train_dataset)
    # 2. get the records with the same BB count
    bb_count = 1
    if bb_count not in bulk_len_record.keys():
        raise ValueError("bb_count not in bulk_len_record")
    records_with_same_bb = bulk_len_record[bb_count]
    new_dataset = []
    min_num_inst, max_num_inst = 1, 10
    # 3. Get the records with the same instruction count, currently we only filter the records with instruction count from 1 to 9
    for inst_count in range(min_num_inst, max_num_inst):
        record_list = [record for record in records_with_same_bb if np.sum(record['llvm_ir']['bb_count']['bb_list_size']) == inst_count]
        type_list = get_record_list_from_bb_count(record_list)
        for same_type_list in type_list:
            new_dataset.extend([same_type_list[0][0] for same_type_list in type_list])
    # 4. Get the records with the instruction count greater than 10
    new_dataset.extend([
        record for record in bulk_len_record[bb_count] 
        if np.sum(record['llvm_ir']['bb_count']['bb_list_size']) >= max_num_inst
    ])
    for bb_count, records in bulk_len_record.items():
        if bb_count == 1:
            continue
        for record in records:
            new_dataset.append(record)
    
    filtered_dataset = Dataset.from_list(new_dataset)
    filtered_dataset.save_to_disk(result_path)

        
if __name__ == "__main__":
    fire.Fire(filter_by_llvm_diff)
