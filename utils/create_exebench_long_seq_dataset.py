from datasets import load_from_disk, concatenate_datasets

part_list = [2, 3, 4, 5, 6, 7]

dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_{}_llvm_extract_func_ir_assembly_O2_llvm_diff"
a = load_from_disk(dataset_path.format(0))

dataset_list = [load_from_disk(dataset_path.format(part)) for part in part_list]
for d in dataset_list:
    print(d.features.type)
    print()
# assert(dataset_list[0].features.type == dataset_list[1].features.type)
# exebench_dataset = concatenate_datasets(dataset_list)
# print(len(exebench_dataset))
