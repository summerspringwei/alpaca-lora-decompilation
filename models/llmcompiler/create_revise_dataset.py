import os
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
    target_ass_file_path = os.path.join(folder_path, "target.s")

    error_msg = read_file_content(error_file_path)
    predict_ir = read_file_content(predict_file_path)
    target_ir = read_file_content(target_ir_file_path)
    target_ass = read_file_content(target_ass_file_path)
    
    ass = preprocessing_assembly.preprocessing_assembly(target_ass)
    target_ir = preprocessing_llvm_ir.preprocessing_llvm_ir(target_ir)
    error_msg = preprocessing_llc_error_msg.preprocessing_llc_error_msg(error_msg)

    return (ass, predict_ir, error_msg, target_ir)


def create_from_error_file_list(file_path: str):
    error_type_dict = analyze_error_type.analyze_error_type(file_path)
    # Read the error prediction results
    file_list = error_type_dict["use of undefined value"]
    for file in file_list:
        process_file(os.path.dirname(file))


if __name__ == "__main__":
    create_from_error_file_list("analysis/tmp_all_error_predict_list.txt")
