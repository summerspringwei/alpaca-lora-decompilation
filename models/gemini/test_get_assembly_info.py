
import tempfile
from datasets import load_from_disk
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly, extract_wrapper_assembly_functions
from models.gemini.parse_assembly import extract_called_functions
from utils.extract_asm_function_define import split_elf_functions
dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"

dataset = load_from_disk(dataset_dir_path)
for record in dataset:
    # print(record)
    # 1. Extract the called functions from the assembly code
    function_list = extract_called_functions(record['asm']['code'][-1])
    if len(function_list) == 0:
        continue
        # print("Function list:", function_list)
    # predict_execution_success = eval_assembly(record, record['asm']['code'][-1])
    if len(function_list) == 0:
        continue
    # 2. Extract the assembly code of the called functions
    functions_dict = extract_wrapper_assembly_functions(record)
    # print("Functions dict:", functions_dict.keys())
    for func_name in function_list:
        if func_name.find("@PLT") != -1:
            func_name = func_name.split("@PLT")[0]
        if func_name in functions_dict:
            print(f"Function {func_name} found")
            print(functions_dict[func_name])
        else:
            print(f"Function {func_name} not found in assembly.")
        print("==" * 20)
    assert("main" in functions_dict.keys())

    # 3. Get the function name of the decompiled function
    asm_func_name = split_elf_functions(record['asm']['code'][-1]).keys()[0]

    # 4. Get how the decompiled function is called
    

    # if predict_execution_success:
    #     print("Execution success")
    # else:
    #     print("Execution failed")
