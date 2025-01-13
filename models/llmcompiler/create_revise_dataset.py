import os

from datasets import Dataset, load_from_disk, DatasetDict
from analysis import analyze_error_type
from utils import preprocessing_assembly, preprocessing_llvm_ir, preprocessing_llc_error_msg


def read_file_content(file_path: str) -> str:
    with open(file_path, "r") as f:
        content = f.read()
    return content


def process_file(folder_path: str) -> tuple[str, str, str, str]:
    """"Read from our exebench folder and return the assembly, predicted IR, error message, and target IR.
    
    """
    error_file_path = os.path.join(folder_path, "error_predict.error")
    predict_file_path = os.path.join(folder_path, "predict.ll")
    target_ir_file_path = os.path.join(folder_path, "target.ll")
    assembly_file_path = os.path.join(folder_path, "target.s")

    error_msg = read_file_content(error_file_path)
    predict_ir = read_file_content(predict_file_path)
    target_ir = read_file_content(target_ir_file_path)
    assembly = read_file_content(assembly_file_path)
    
    assembly = preprocessing_assembly.preprocessing_assembly(assembly)
    target_ir = preprocessing_llvm_ir.preprocessing_llvm_ir(target_ir)
    error_msg = preprocessing_llc_error_msg.preprocessing_llc_error_msg(error_msg)

    return {"assembly": assembly, "predict_ir": predict_ir, "error_msg": error_msg, "target_ir": target_ir}


def create_from_error_file_list(file_path_list: list[str]):
    error_type_dict = analyze_error_type.get_error_from_list_files(file_path_list)
    # error_type_dict = analyze_error_type.analyze_error_type(file_path)
    # Read the error prediction results
    file_list = error_type_dict["use of undefined value"]
    output = [process_file(os.path.dirname(file)) for file in file_list]
    
    return output


def main():
    output = create_from_error_file_list([
        "/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_list0.txt", 
        "/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_list3.txt", 
        "/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_list4.txt", 
        "/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_list5.txt", 
        "/home/xiachunwei/Projects/alpaca-lora-decompilation/analysis/tmp_all_error_predict_list6.txt", 
    ])
    instruction = "Give the assembly code, predicted IR and the error message, generate the target IR code."
    input_str = "assembly code: <code> {assembly} </code>, predicted IR: <code> {predict_ir} </code>, error message: {error_msg} "
    label = "target IR: {target_ir}"
    instruction_list = [{
        "instruction": instruction, 
        "input": input_str.format(
            assembly=sample["assembly"], 
            predict_ir=sample["predict_ir"], 
            error_msg=sample["error_msg"]),
        "output": label.format(target_ir=sample["target_ir"])} for sample in output]
    # We need to format the dataset to (instruction: xxx, input: assembly + predict_ir + error_msg, label: target_ir)
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(instruction_list)
    })
    dataset_dict.save_to_disk("/home/xiachunwei/Datasets/revise_exebench/revised_exebench_split_03456")
    dataset = load_from_disk("/home/xiachunwei/Datasets/revise_exebench/revised_exebench_split_03456")
    print(dataset)


if __name__ == "__main__":
    main()
