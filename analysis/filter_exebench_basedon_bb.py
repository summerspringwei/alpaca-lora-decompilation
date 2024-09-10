path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_2_llvm_extract_func_ir_assembly_O2"
bb_count_binary: str = "/home/xiachunwei/Projects/llm4compiler/src/cpp/build/count_llvm_ir_bb"

import tqdm
import subprocess
import tempfile
import numpy as np

from datasets import load_from_disk

from analysis.count_bb import set_bb_count, get_bulk_list

# llvm_diff = "/home/xiachunwei/Software/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04/bin//llvm-mdiff"
llvm_diff = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-mdiff"

def write_to_tmp_file(code: str):
    f = tempfile.NamedTemporaryFile(delete=True, mode='w')
    f.write(code)
    f.flush()
    # f.close()
    return f


def analyze_the_diff(diff: str)->bool:
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


def test():
    train_dataset = load_from_disk(
        path_to_dataset
    )
    train_dataset = train_dataset.map(set_bb_count, num_proc=8)
    bulk_len_record = get_bulk_list(train_dataset)
    # Get the records with the same bb count
    bb_count = 1
    if bb_count not in bulk_len_record.keys():
        raise ValueError("bb_count not in bulk_len_record")
    records_with_same_bb = bulk_len_record[bb_count]
    # Get the records with the same instruction count
    inst_count = 8
    record_list = [record for record in records_with_same_bb if np.sum(record['llvm_ir']['bb_count']['bb_list_size']) == inst_count]
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
            # diff = cmd_out.stderr.decode("utf-8")
            # print(diff)
            # is_diff = analyze_the_diff(diff)
            # if not is_diff:
            #     current_type_list.append((record, f2))
            #     find_same_type = True
            #     break
        if not find_same_type:
            type_list.append([(record, f2), ])
    
    print(f"total {len(record_list)} alltypes {len(type_list)}")
    type_list.sort(key=lambda x: len(x), reverse=True)
    for type in type_list:
        print(";"*20)
        print(";",len(type))
        print(type[0][0]['llvm_ir']['code'][-1])



        
if __name__ == "__main__":
    test()
# ['/home/xiachunwei/Software/clang+llvm-17.0.2-x86_64-linux-gnu-ubuntu-22.04/bin//llvm-diff', '/tmp/tmpuecl71xl', '/tmp/tmpl3zrwidm']