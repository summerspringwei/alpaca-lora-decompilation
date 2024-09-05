
import os
import re
import json
import subprocess
import logging
import pathlib
from typing import Dict
from multiprocessing import Pool
from datasets import load_from_disk
from exebench import Wrapper, diff_io, exebench_dict_to_dict, LLVMAssembler

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)s - %(message)s ', level=logging.INFO)


def extract_function_names_from_llvm_ir(llvm_ir_code):
    # Regular expression to match function definitions in LLVM IR and capture function names
    # function_pattern = re.compile(r'define\s+\w+\s+@(\w+)\s*\([^)]*\)\s*{')
    # function_pattern = re.compile(r'define\s+\S+\s+@(\w+)\s*\(')
    function_pattern = re.compile(r'define\s+(?:\S+\s+)*@\s*([\w\d_]+)\s*\(')

    # Find all function names
    function_names = function_pattern.findall(llvm_ir_code)
    if function_names and len(function_names) > 0:
        return function_names[0]
    else:
        return None


def extract_function_name_from_C(declaration):
    """
    Extracts the function name from a C++ or C function declaration.

    Args:
    declaration (str): A string containing the function declaration.

    Returns:
    str: The name of the function, or None if no function name is found.
    """
    # Regular expression pattern to match C++ and C function declarations
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(const)?\s*(;)?\s*$'
    
    # Search for the pattern in the declaration
    match = re.search(pattern, declaration)
    
    if match:
        return match.group(1)
    else:
        return None


def compile_predicted_record(record: Dict, validation_dir: str = "./tmp_validate_exebench")->bool:
    """Compile the llvm ir to assembly and save the results to the validation directory, return true if success compile"""
    predict_llvm_ir = record['predict']
    target_llvm_ir = record['output']
    file_path = record["file"]
    full_path = os.path.join(validation_dir, file_path)
    # 1. First save LLVM IR and assembly to file
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    predict_llvm_ir_path = os.path.join(full_path, "predict.ll")
    predict_assembly_path = os.path.join(full_path, "predict.s")
    target_llvm_ir_path = os.path.join(full_path, "target.ll")
    target_assembly_path = os.path.join(full_path, "target.s")
    predict_error_path = os.path.join(full_path, "error_predict.error")

    with open(predict_llvm_ir_path, 'w') as f:
        f.write(predict_llvm_ir)
    with open(target_llvm_ir_path, 'w') as f:
        f.write(target_llvm_ir[0] if isinstance(target_llvm_ir, list) else target_llvm_ir)
    predict_success, target_success = True, True
    try:
        # 2. Compile predicted llvm ir to assembly
        cmd = ["llc", predict_llvm_ir_path, "-o", predict_assembly_path]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            # Save the stderr output to the specified file
            with open(predict_error_path, 'w') as f:
                f.write(ret.stderr.decode())
            predict_success = False
    except Exception as e:
        logging.error(e)
        predict_success = False
    try:
        # 3. Compile the ground truth llvm ir to assembly
        cmd = ["llc", target_llvm_ir_path, "-o", target_assembly_path]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            # Save the stderr output to the specified file
            with open(predict_error_path, 'a') as f:
                f.write(ret.stderr.decode())
            target_success = False
    except Exception as e:
        logging.error(e)
        target_success = False
    return predict_success, predict_assembly_path, target_success, target_assembly_path


def eval_assembly(row: Dict, assembly: str) -> bool:
    success = True
    synth_wrapper = None
    try:
        c_deps=(row['synth_deps'] + '\n' +
                    row['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
                        'typedef int bool;', '')
        synth_wrapper = Wrapper(
            c_deps=c_deps + '\n',
            func_c_signature=row['func_head_types'].replace('extern', ''),
            func_assembly=assembly,
            cpp_wrapper=row['synth_exe_wrapper'],
            assembler_backend=LLVMAssembler())
        count, total = 0, len(row['synth_io_pairs']['input'])
        for i, o in zip(row['synth_io_pairs']['input'],
                        row['synth_io_pairs']['output']):
            observed_output = synth_wrapper(
                exebench_dict_to_dict(i))  # Run synthetic
            if observed_output is None:
                logging.error('Error: The code could not be compiled')
                success = False
                return success
            # print(observed_output, exebench_dict_to_dict(o))
            count += 1 if diff_io(
                observed_output=observed_output,
                expected_output=exebench_dict_to_dict(o)) else 0
        success = (count == total)
        if not success:
            logging.info(
                f"Error for {row['path']} total cases {total}, success cases {count}"
            )
    except Exception as e:
        logging.error(f"Error for {row['path']}")
        logging.error(e)
        success = False
    finally:
        return success



def validate_by_execution(record: Dict, row: Dict)->Dict:
    matched_predict_llvm_ir = extract_llmcompiler_code_blocks(record["predict"])
    if matched_predict_llvm_ir and len(matched_predict_llvm_ir) > 0:
        record["predict"] = matched_predict_llvm_ir[0]
    predict_success, predict_assembly_path, target_success, target_assembly_path = compile_predicted_record(record)
    predict_execution_success, target_execution_success = False, False
    # Validate the predict assembly
    if predict_success:
        try:
            with open(predict_assembly_path, 'r') as f:
                predict_execution_success = eval_assembly(row, f.read())
        except Exception as e:
            logging.error(e)
            predict_execution_success = False
    # Validate the target assembly
    if target_success:
        try:
            with open(target_assembly_path, 'r') as f:
                target_execution_success = eval_assembly(row, f.read())
        except Exception as e:
            logging.error(e)
            target_execution_success = False
    record["predict_compile_success"] = predict_success
    record["predict_execution_success"] = predict_execution_success
    record["target_compile_success"] = target_success
    record["target_execution_success"] = target_execution_success

    print((predict_success, predict_execution_success, target_success, target_execution_success))
    # return (predict_success, predict_execution_success, target_success, target_execution_success)
    return record


def wrapper(args):
    if len(args) != 2 or not isinstance(args[0], dict) or not isinstance(args[1], dict):
        logging.error(f"Invalid input: {args}")
        return None
    return validate_by_execution(*args)


def extract_llmcompiler_code_blocks(text):
    pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    matches = pattern.findall(text)
    return matches





def validate_exebench(path_to_json: str, path_to_dataset: str):
    dataset = load_from_disk(
        path_to_dataset
    )
    path_to_row_mapping = {}
    for row in dataset:
        if row["path"] not in path_to_row_mapping.keys():
            path_to_row_mapping[row['path']] = [row,]
        else:
            path_to_row_mapping[row['path']].append(row)
            logging.info(f"Duplicate path: {row['path']}")
    
    all_records = json.load(open(path_to_json, 'r'))
    path_to_record_mapping = {}
    for record in all_records:
        # Preprocessing the LLM output here:
        if record["predict"].find("code") >= 0:
            matched_predict_llvm_ir = extract_llmcompiler_code_blocks(record["predict"])
            if matched_predict_llvm_ir and len(matched_predict_llvm_ir) > 0:
                record["predict"] = matched_predict_llvm_ir[0]
            else:
                logging.error(f"Cannot find code block in {record['predict']}")
        path_to_record_mapping[record['file']] = record
    
    # We need also to make sure the function name is the same
    path_to_record_row_mapping = {}
    for record in all_records:
        record_func = record['func_head_types']
        if record['file'] in path_to_row_mapping:
            for row in path_to_row_mapping[record['file']]:
                row_func = row['func_head_types']
                if record["predict"].find("aarch64-none-linux-gnu") >= 0:
                    logging.info(f"Predict wrong arch type for {row['path']}:{row['func_head_types']}")
                    continue
                if record_func == row_func:
                    path_to_record_row_mapping[record['file']] = (path_to_record_mapping[record['file']], row)
                    break
        else:
            logging.error(f"Cannot find record for {record['file']}")

    args = [value for _, value in path_to_record_row_mapping.items()]
    # for record, row in args:
    #     validate_by_execution(record, row)
    with Pool(processes=40) as pool:
        results = pool.map(wrapper, args)

    predict_compile_results = [r["predict_compile_success"] if isinstance(r, dict) else False for r in results]
    predict_execution_results = [r["predict_execution_success"] if isinstance(r, dict) else False for r in results]
    target_compile_results = [r["target_compile_success"] if isinstance(r, dict) else False for r in results]
    target_execution_results = [r["target_execution_success"] if isinstance(r, dict) else False for r in results]
    logging.info(f"""Total records: {len(all_records)}, 
                 predict_compile_success:{sum(predict_compile_results)}, 
                 predict_execution_success: {sum(predict_execution_results)},
                 target_compile_success: {sum(target_compile_results)},
                 target_execution_success: {sum(target_execution_results)}""")
    json.dump(results, open("exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd_validate_exebench.json", 'w'), indent=4, sort_keys=False, separators=(',', ':'))


if __name__ == "__main__":
    # path_to_json = "bart_exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_json = "deepseek-coder-1.3b-exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_dataset = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_ir_assembly_O2"
    # path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json"
    path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json"
    path_to_dataset = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"
    validate_exebench(path_to_json, path_to_dataset)
