
import os
import json
import subprocess
import logging
import pathlib
from typing import Dict
from multiprocessing import Pool
from datasets import load_dataset, load_from_disk

from exebench import (eval_assembly)

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)s - %(message)s ', level=logging.INFO)


def compile_predicted_record(record: Dict, validation_dir: str = "./validate_exebench")->bool:
    """Compile the llvm ir to assembly and save the results to the validation directory, return true if success compile"""
    predict_llvm_ir = record['predict']
    target_llvm_ir = record['output']
    file_path = record["file"]
    full_path = os.path.join(validation_dir, file_path)
    success = True
    try:
        # 1. First save results
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        predict_llvm_ir_path = os.path.join(full_path, "predict.ll")
        predict_assembly_path = os.path.join(full_path, "predict.s")
        target_llvm_ir_path = os.path.join(full_path, "target.ll")
        target_assembly_path = os.path.join(full_path, "target.s")
        predict_error_path = os.path.join(full_path, "error_predict.error")

        with open(predict_llvm_ir_path, 'w') as f:
            f.write(predict_llvm_ir)
        with open(target_llvm_ir_path, 'w') as f:
            f.write(target_llvm_ir)
            
        # 2. Compile llvm ir to assembly
        cmd = ["llc", predict_llvm_ir_path, "-o", predict_assembly_path]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            # Save the stderr output to the specified file
            with open(predict_error_path, 'w') as f:
                f.write(ret.stderr.decode())
            success = False
    except Exception as e:
        logging.error(e)
        success = False
    return success, predict_assembly_path


def validate_by_execution(record: Dict, row: Dict)->bool:
    compile_success, predict_assembly_path = compile_predicted_record(record)
    execution_success = False
    if compile_success:
        # 3. Validate the assembly
        try:
            with open(predict_assembly_path, 'r') as f:
                execution_success = eval_assembly(row, f.read())
        except Exception as e:
            logging.error(e)
            execution_success = False
    print((compile_success, execution_success))
    return (compile_success, execution_success)


def wrapper(args):
    if len(args) != 2 or not isinstance(args[0], dict) or not isinstance(args[1], dict):
        logging.error(f"Invalid input: {args}")
        return False
    return validate_by_execution(*args)


def validate_exebench(path_to_json: str, path_to_dataset: str):
    dataset = load_from_disk(
        path_to_dataset
    )
    path_to_row_mapping = {}
    for row in dataset:
        path_to_row_mapping[row['path']] = row
    

    all_records = json.load(open(path_to_json, 'r'))
    path_to_record_mapping = {}
    all_records = all_records[0:1000]
    for record in all_records:
        path_to_record_mapping[record['file']] = record
    
    for record in all_records:
        if record['file'] in path_to_row_mapping:
            path_to_record_mapping[record['file']] = (path_to_record_mapping[record['file']], path_to_row_mapping[record['file']])
        else:
            logging.error(f"Cannot find record for {record['file']}")
    
    args = [value for _, value in path_to_record_mapping.items()]
    with Pool(processes=40) as pool:
        results = pool.map(wrapper, args)
    compile_results = [r[0] if isinstance(r, tuple) else False for r in results]
    execution_results = [r[1] if isinstance(r, tuple) else False for r in results]
    logging.info(f"Total records: {len(all_records)}, compile_success:{sum(compile_results)}, success: {sum(execution_results)}")




if __name__ == "__main__":
    path_to_json = "validation_decompilation/result_deepseek-coder_train_real_simple_io-llvm-assembly-batch.json"
    path_to_dataset = "/data/xiachunwei/Datasets/exebench/train_real_simple_io-llvm-assembly-batch-clang-15"
    validate_exebench(path_to_json, path_to_dataset)
