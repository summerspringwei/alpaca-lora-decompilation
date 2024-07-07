
import os
import json
import subprocess
import logging
import pathlib
from typing import Dict
from multiprocessing import Pool
from datasets import load_from_disk
from exebench import Wrapper, diff_io, exebench_dict_to_dict, LLVMAssembler

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)s - %(message)s ', level=logging.INFO)


def compile_predicted_record(record: Dict, validation_dir: str = "./validate_exebench")->bool:
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

# def validate(row: Dict):
#     success = False
#     try:
#         synth_wrapper = Wrapper(
#             c_deps=(row['synth_deps'] + '\n' +
#                     row['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
#                         'typedef int bool;', ''),
#             func_c_signature=row['func_head_types'].replace('extern', ''),
#             func_assembly=row['asm']['code'][0],
#             cpp_wrapper=row['synth_exe_wrapper'],
#             # assembler_backend=LLVMAssembler()
#         )
#         observed_output = synth_wrapper(
#             exebench_dict_to_dict(row['synth_io_pairs']['input']
#                                   [0]))  # Run synthetic example number 0
#         success = True if diff_io(
#             observed_output=observed_output,
#             expected_output=exebench_dict_to_dict(
#                 row['synth_io_pairs']['output'][0])) else False
#     except Exception as e:
#         # Very occasionally the compilating using func_assembly=row['asm']['code'][0] seems to fail.
#         # My best guess at this moment is that the self-contained function assembly is not "self-contained enough"
#         # in a few cases, and in these cases it's better to recompile everything and run it all together.
#         # TODO: fix, or find a better explanation
#         # pass
#         logging.error(f'Error: {e}')
#         return False
#     return success


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



def validate_by_execution(record: Dict, row: Dict)->bool:
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
    
    print((predict_success, predict_execution_success, target_success, target_execution_success))
    return (predict_success, predict_execution_success, target_success, target_execution_success)


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
    all_records = all_records[:100]
    for record in all_records:
        path_to_record_mapping[record['file']] = record
    
    for record in all_records:
        if record['file'] in path_to_row_mapping:
            a= path_to_record_mapping[record['file']]
            b = path_to_row_mapping[record['file']]
            path_to_record_mapping[record['file']] = (a, b)
            # path_to_record_mapping[record['file']] = (path_to_record_mapping[record['file']], path_to_row_mapping[record['file']])
        else:
            logging.error(f"Cannot find record for {record['file']}")

    args = [value for _, value in path_to_record_mapping.items()]
    for record, row in args:
        if record["file"]=="pascalorama/megadrive-studio/gp2/emu/m68k/m68kopac.c":
            validate_by_execution(record, row)
    with Pool(processes=40) as pool:
        results = pool.map(wrapper, args)
    predict_compile_results = [r[0] if isinstance(r, tuple) else False for r in results]
    predict_execution_results = [r[1] if isinstance(r, tuple) else False for r in results]
    target_compile_results = [r[2] if isinstance(r, tuple) else False for r in results]
    target_execution_results = [r[3] if isinstance(r, tuple) else False for r in results]
    logging.info(f"""Total records: {len(all_records)}, 
                 compile_success:{sum(predict_compile_results)}, 
                 success: {sum(predict_execution_results)},
                 target: {sum(target_execution_results)}""")


if __name__ == "__main__":
    # path_to_json = "bart_exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_json = "deepseek-coder-1.3b-exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_dataset = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_ir_assembly_O2"

    path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json"
    path_to_dataset = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2"
    validate_exebench(path_to_json, path_to_dataset)
