
import os
import json
import subprocess
import re
from pathlib import Path
import fire

def dump_llvm_module_to_cfg(file_path: str):
    try:
        cmd_out = subprocess.run(["opt", "-dot-cfg", "-disable-output", "-enable-new-pm=0", file_path], capture_output=True, text=True)
        if cmd_out.returncode == 0:
            # Get the function name
            match = re.search(r"'(.*?)'", cmd_out.stderr)
            if match:
                function_name = match.group(1)
                graph_file_name = file_path.replace(".ll", ".png")
                Path(os.path.dirname(graph_file_name)).mkdir(parents=True, exist_ok=True)
                cmd_out = subprocess.run(["dot", "-Tpng", f"{function_name}", "-o", graph_file_name])
                if cmd_out.returncode == 0:
                    print(f"sunccesfully generate {graph_file_name}")
                    return 1
        else:
            print("predict failed", file_path, cmd_out.stderr)
    except Exception as e:
            print(e)
    return 0


def verify_perfect_decompilation(predict_file: str, output_file: str):
    # 1. Compare the content of two llvm ir
    perfect = False
    with open(predict_file, "r") as f:
        predict_content = f.read()
    with open(output_file, "r") as f:
        output_content = f.read()
    if predict_content == output_content:
        perfect = True
    # 2. If not perfect, compare the assembly code
    predict_asm = subprocess.run(["llc", "-march=x86-64", "-filetype=asm", predict_file], capture_output=True, text=True)
    output_asm = subprocess.run(["llc", "-march=x86-64", "-filetype=asm", output_file], capture_output=True, text=True)
    if predict_asm.returncode == 0 and output_asm.returncode == 0:
        with open(predict_file.replace(".ll", ".s"), "r") as f:
            predict_asm_content = f.read()
        with open(output_file.replace(".ll", ".s"), "r") as f:
            output_asm_content = f.read()
        if predict_asm_content == output_asm_content:
            perfect = True
    # TODO(Chunwei): Add more verification methods
    return perfect



def main(val_file_path: str = "val.json", out_dir: str = "val_result"):
    programs = json.load(open(val_file_path, 'r'))
    
    success_compile, perfect_compile = 0, 0
    for record in programs:
        dir_path = os.path.join(out_dir, record["file:"].rstrip(".ll"))
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_path, "predict.ll"), "w") as f:
            content = record["predict"]
            if content.find("</s><s>"):
                content = content.replace("</s><s>", "\n")
            f.write(content)
        with open(os.path.join(dir_path, "output.ll"), "w") as f:
            f.write(record["output"])
        # opt -dot-cfg -disable-output -enable-new-pm=0 file.ll
        ret_code = dump_llvm_module_to_cfg(os.path.join(dir_path, "predict.ll"))
        success_compile += ret_code
        dump_llvm_module_to_cfg(os.path.join(dir_path, "output.ll"))
        if ret_code == 1:
            if verify_perfect_decompilation(os.path.join(dir_path, "predict.ll"), os.path.join(dir_path, "output.ll")):
                perfect_compile += 1
    
    print(f"Total programs: {len(programs)}, of which {perfect_compile} are perfectly decompiled, {success_compile} are successfully compiled.")


def test_dump():
    file_path = "decompilation_val/test.ll"
    dump_llvm_module_to_cfg(file_path)


if __name__ == "__main__":
    fire.Fire(main)
