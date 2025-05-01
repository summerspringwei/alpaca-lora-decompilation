import subprocess
import logging
import os
import json
import re
import pickle
import logging
from openai import OpenAI
# from google import genai
from functools import partial
from datasets import load_from_disk, Dataset
from multiprocessing import Pool
import sys
print("Python Path:", sys.path)  # Check if project root is listed
print("Current Working Directory:", os.getcwd())  # Ensure you're in the right dir
from utils.evaluate_exebench import compile_llvm_ir
from utils.preprocessing_assembly import preprocessing_assembly
from utils.openai_helper import extract_llvm_code_from_response, format_compile_prompt, format_execution_prompt

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3",timeout=1800)
model_name = 'ep-20250317013717-m9ksl'


def huoshan_deepseek_r1(client, prompt: str):
    response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
    return response


def run_command(command, cwd):
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            logging.warning(f"Command failed: {' '.join(command)}\nError: {result.stderr}")
        return result.returncode

def parse_test_results(output):
    result_dict = {}
    pattern = r"# (\w+):\s+(\d+)"
    matches = re.findall(pattern, output)
    for key, value in matches:
        result_dict[key] = int(value)
    return result_dict


def decompilation_loop(client, func_name: str, output_dir, num_retry: int = 10, remove_comments: bool = True, cwd="/home/xiachunwei/Projects/coreutils"):
    with open(os.path.join(output_dir, f"src/tail_target_{func_name}.s"), 'r') as f:
        asm_code = f.read()
    asm_code = preprocessing_assembly(asm_code, remove_comments=remove_comments)
    input_str =f"""Please decompile the following assembly code to LLVM IR and please place the final generated LLVM IR code between ```llvm and ```: {asm_code} \n Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once."""
    prompt = input_str.format(asm_code=asm_code)
    count = 0
    predict_compile_success, predict_execution_success = False, False
    while count < num_retry and (not predict_compile_success or not predict_execution_success):
        count += 1
        logger.info(f"Retrying {count} times for {func_name}")
        try:
            # 0. Get response from the model
            # response = huoshan_deepseek_r1(client, prompt)
            response_file_path = os.path.join(output_dir, f"response_{func_name}_retry_{count}.pkl")
            response = pickle.load(open(response_file_path, "rb"))
            # pickle.dump(response, open(response_file_path, "wb"))
            predict_llvm_ir = extract_llvm_code_from_response(response)
            if len(predict_llvm_ir) == 0:
                logger.warning(f"Empty prediction for {func_name} on retry {count}")
                continue
            # 1. Try to compile the decompiled LLVM IR with `llc`
            name_hint = f"src/tail_predict_{func_name}"
            compile_success, assembly_path = compile_llvm_ir(predict_llvm_ir, output_dir, name_hint)
            error_msg = ""
            if not compile_success:
                assert(os.path.exists(os.path.join(output_dir, f"{name_hint}.error")))
                with open(os.path.join(output_dir, f"{name_hint}.error"), 'r') as f:
                    error_msg = f.read()
                logger.warning(f"Compilation failed for {func_name} on retry {count}: {error_msg}")
                continue
            # 3. try to link the decompiled ir with the original one
            # link and compile
            cmd = ["llvm-link", f"src/tail_no_{func_name}.ll", f"src/tail_{func_name}.ll", "-o", "src/tail_predict.ll"]
            with open("tmp.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")
                f.write(f"cd {cwd}\n")
                f.write(" ".join(cmd) + "\n")
            os.chmod("tmp.sh", 0o755)
            cmd = ["bash", "tmp.sh"]
            results = subprocess.run(cmd, capture_output=True, text=True)

            if results.returncode != 0:
                logger.warning(f"Error linking files: {results.stderr}")
                error_msg = results.stderr
                continue
            # 4. Compile the linked LLVM IR to object file
            target_object_file = "src/tail_predict.o"
            cmd = ["clang", "-c", "src/tail_predict.ll", "-o", target_object_file]
            results = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
            if results.returncode != 0:
                logger.warning(f"Error compiling files: {results.stderr}")
                error_msg = results.stderr
                continue
            # 5. Link the object file to the executable binary
            run_command([
                "clang", "-Wno-format-extra-args", "-Wno-implicit-const-int-float-conversion",
                "-Wno-tautological-constant-out-of-range-compare", "-g", "-O2", "-Wl,--as-needed",
                "-o", "src/tail", target_object_file, "src/iopoll.o", "src/libver.a",
                "lib/libcoreutils.a", "lib/libcoreutils.a", "-ldl"
            ], cwd)
            # 6. Run the test
            cmd = ["make", "check", "TESTS=\"$(make listtests | tr ' ' '\\n' | grep '^tests/tail')\""]
            with open("tmp.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")
                f.write(f"cd {cwd}\n")
                f.write(" ".join(cmd) + "\n")
            os.chmod("tmp.sh", 0o755)
            cmd = ["bash", "tmp.sh"]
            results = subprocess.run(cmd, capture_output=True, text=True)
            print(results.stdout)
            # 7. Check the test result
            test_result = parse_test_results(results.stdout)
            if test_result["TOTAL"] == test_result["PASS"] + test_result["SKIP"]:
                predict_compile_success = True
                predict_execution_success = True
                logger.info(f"Decompilation successful for {func_name} on retry {count}")
                break
            else:
                prompt = format_compile_prompt(asm_code, predict_llvm_ir, error_msg)

        except Exception as e:
            logging.warning(f"Error during decompilation: {e}")


def auto_test_one_function(func_name, cwd = "/home/xiachunwei/Projects/coreutils"):    
    # Step 1: Extract one function from LLVM IR file
    if run_command(["llvm-extract", f"--func={func_name}", "-S", "src/tail.ll", "-o", f"src/tail_target_{func_name}.ll"], cwd) != 0:
        return

    # Step 2: Delete one function from module
    if run_command(["llvm-extract", "-delete", f"--func={func_name}", "-S", "src/tail.ll", "-o", f"src/tail_no_{func_name}.ll"], cwd) != 0:
        return

    # Step 3: Compile function LLVM IR to assembly
    if run_command(["llc", f"src/tail_target_{func_name}.ll", "-o", f"src/tail_target_{func_name}.s"], cwd) != 0:
        return

    # Step 4: Use the LLM to decompile assembly to IR
    # Placeholder for LLM decompilation logic (not specified in the prompt)
    decompilation_loop(client, func_name, cwd, num_retry=3, remove_comments=True, cwd=cwd)


def test():
    auto_test_one_function("pretty_name")


if __name__ == "__main__":
    test()