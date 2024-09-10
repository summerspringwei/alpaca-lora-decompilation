
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

import logging
logger = logging.getLogger(__name__)


from analysis.count_bb import set_bb_count, get_bulk_list

path_to_dataset = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_2_llvm_extract_func_ir_assembly_O2"

def draw_distribution(path_to_dataset: str):
    train_dataset = load_from_disk(
        path_to_dataset
    )
    train_dataset = train_dataset.map(set_bb_count, num_proc=8)
    bulk_len_record = get_bulk_list(train_dataset)

    # Draw the distribution of number of records by basic block count
    bb_result = [(bb_count, len(records)) for bb_count, records in bulk_len_record.items() if len(records) > 2]
    bb_result.sort(key=lambda x: x[0])
    print(bb_result)
    plt.figure(figsize=(10, 6))
    # bars = plt.bar(bb_count_list, num_records_list, color='blue')
    bars = plt.bar([x[0] for x in bb_result], [x[1] for x in bb_result], color='blue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

    plt.xlabel('Basic Block Count')
    plt.ylabel('Number of Records')
    plt.title('Distribution of Number of Records by Basic Block Count')
    plt.grid(True)
    plt.savefig("bb_count_distribution.png")
    plt.close()

    # Get the distribution of number of bb = 1
    num_bb = 1
    inst_dict = {}
    for record in bulk_len_record[num_bb]:
        inst_count = np.sum(record['llvm_ir']['bb_count']['bb_list_size'])
        if inst_count not in inst_dict:
            inst_dict[inst_count] = 1
        else:
            inst_dict[inst_count] += 1
    inst_result = [(inst_count, num_inst_list) for inst_count, num_inst_list in inst_dict.items() if num_inst_list > 4]
    inst_result.sort(key=lambda x: x[0])
    plt.figure(figsize=(10, 6))
    bars = plt.bar([x[0] for x in inst_result], [x[1] for x in inst_result], color='blue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

    print(inst_result)
    plt.xlabel('Instruction Count')
    plt.ylabel('Number of Records')
    plt.title(f'Distribution of Number of Records by Instruction Count with BB={num_bb}')
    plt.grid(True)
    plt.savefig("inst_count_distribution.png")

    for record in train_dataset:
        if np.sum(record['llvm_ir']['bb_count']['bb_list_size']) == 3:
            print(record['llvm_ir']['code'][-1])
    

def main():
    draw_distribution(path_to_dataset)

if __name__ == "__main__":
    main()
