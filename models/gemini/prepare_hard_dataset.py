import os
import json
import subprocess
import tempfile
import fire
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
from utils.exebench_dataset_processing import filter_cannot_parse, map_func_info, filter_has_loops, filter_unused_args, filter_with_struct_args, filter_num_of_instructions, dump_llvm_ir

def main(
    dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff",
    saved_dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_hard_sample_100",
    bb_threshold = 3
):
    exebench_data = load_from_disk(dataset_path)
    print("exebench_data", len(exebench_data))
    # First map to add functions field
    exebench_data = exebench_data.filter(filter_cannot_parse, num_proc=16, load_from_cache_file=True)
    print("exebench_data after filter_cannot_parse", len(exebench_data))
    dataset_with_func_info = exebench_data.map(map_func_info, num_proc=16, load_from_cache_file=False)
    print("dataset_with_func_info", len(dataset_with_func_info))
    dataset_with_loops = dataset_with_func_info.filter(filter_has_loops, num_proc=16)
    print("dataset_with_loops", len(dataset_with_loops))

    dataset_filter_unused_args = dataset_with_func_info.filter(filter_unused_args,
                                                            num_proc=16)
    print("dataset_filter_unused_args", len(dataset_filter_unused_args))
    # Then filter based on functions length
    dataset_with_called_functions = dataset_filter_unused_args.filter(
        lambda record: len(record['func_info']["functions"][0]["called_functions"]
                        ) > 0,
        num_proc=16)
    print("dataset_with_called_functions", len(dataset_with_called_functions))
    bb_more_than_threshold = dataset_with_called_functions.filter(
        lambda record: record['llvm_ir']['bb_count']["bbcount"] > bb_threshold,
        num_proc=16)
    print("bb_more_than_threshold", len(bb_more_than_threshold))

    dataset_with_struct_args = bb_more_than_threshold.filter(
        filter_with_struct_args,
        num_proc=16)
    print("dataset_with_struct_args", len(dataset_with_struct_args))
    dataset_with_num_of_instructions = dataset_with_struct_args.filter(
        filter_num_of_instructions,
        num_proc=16)
    print("dataset_with_num_of_instructions", len(dataset_with_num_of_instructions))
    # Randomly sample 100 records
    dataset_with_num_of_instructions.save_to_disk(saved_dataset_path)
    dump_llvm_ir(dataset_with_num_of_instructions, "tmp_dir")
    

if __name__ == "__main__":
    fire.Fire(main)
