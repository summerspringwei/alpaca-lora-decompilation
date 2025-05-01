import os
import json

import pickle
import logging
from openai import OpenAI
# from google import genai
from functools import partial
from datasets import load_from_disk, Dataset
from multiprocessing import Pool
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.preprocessing_assembly import preprocessing_assembly
from .openai_helper import extract_llvm_code_from_response, format_compile_prompt, format_execution_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global client
# DeepSeek-R1 on huoshan engine
client = OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3",timeout=1800)
model_name = 'ep-20250317013717-m9ksl'

# GLM-Z1-32B on vllm
# client = OpenAI(api_key="token-llm4decompilation-abc123", base_url="http://localhost:9001/v1",timeout=1800)
# model_name = "/home/xiachunwei/Datasets/Models/GLM-Z1-32B-0414/"

# GPT-4.1 on OpenAI API
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# model_name = "gpt-4.1"

def huoshan_deepseek_r1(client, prompt: str):
    response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
    return response


def huoshan_deepseek_r1_batch(client, prompt_list: list[str]):
    response = client.chat.completions.create(
                model='ep-20250317013717-m9ksl',
                messages=[{"role": "user", "content": prompt} for prompt in prompt_list],
                stream=False
            )
    return response

global huoshan_deepseek_r1_predict
huoshan_deepseek_r1_predict = partial(huoshan_deepseek_r1, client=client)


def prepare_prompt(record, remove_comments: bool = True):
    asm_code = record["asm"]["code"][-1]
    asm_code = preprocessing_assembly(asm_code, remove_comments=remove_comments)
    input_str =f"""Please decompile the following assembly code to LLVM IR and please place the final generated LLVM IR code between ```llvm and ```: {asm_code} \n Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once."""
    prompt = input_str.format(asm_code=asm_code)
    return prompt


# def extract_llvm_code_from_response(response):
#     result = response.choices[0].message.content
#     if "choices" in response and len(response["choices"]) > 0:
#         result = response.choices[0].message.content
#         parts = [choice["message"]["content"] for choice in response["choices"] if
#                     "message" in choice and "content" in choice["message"]]
#     if not parts and result:
#         parts = [result]
#     llvm_code = [extract_llvm_code(part) for part in parts]
#     llvm_code = [code[0] for code in llvm_code if len(code) > 0]
#     predict = llvm_code[0] if len(llvm_code) > 0 else ""
#     if len(llvm_code) == 0:
#         logger.warning("No LLVM code found in the response.")
    
#     return predict




def evaluate_response(response, record, idx, validate_dir):
    predict = extract_llvm_code_from_response(response)
    sample_dir = os.path.join(validate_dir, f"sample_{idx}")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
    
    predict_execution_success, target_execution_success = False, False
    predict_compile_success, predict_assembly_path = compile_llvm_ir(predict, sample_dir, name_hint="predict")
    if predict_compile_success:
        with open(predict_assembly_path, 'r') as f:
            predict_execution_success = eval_assembly(record, f.read())
    target_compile_success, target_assembly_path = compile_llvm_ir(record["llvm_ir"]["code"][-1], sample_dir, name_hint="target")
    if target_compile_success:
        with open(target_assembly_path, 'r') as f:
            target_execution_success = eval_assembly(record, f.read())

    validation_results = {
        "idx": idx,
        "path": record["path"],
        "func_head": record["func_head"],
        "predict_compile_success": predict_compile_success,
        "predict_execution_success": predict_execution_success,
        "target_compile_success": target_compile_success,
        "target_execution_success": target_execution_success,
    }

    return validation_results
           

def process_one(prompt: str, idx: int, output_dir: str, record: dict):
    response = huoshan_deepseek_r1(client, prompt)
    # Make sure first save result to persistant storage
    pickle.dump(response, open(os.path.join(output_dir, f"response_{idx}.pkl"), "wb"))
    # validate the output
    validation_results = {}
    try:
        validation_results = evaluate_response(response, record, idx, output_dir)
    except Exception as e:
        logging.warning(f"Error in evaluating response for index {idx}: {e}")
    return response, validation_results


def LLM_predict_openai_API(dataset: list, output_dir: str, num_processes: int = 8, remove_comments: bool = True):
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    with Pool(processes=num_processes) as pool:
        args = [
            (prepare_prompt(record, remove_comments), idx, output_dir, record) for idx, record in enumerate(dataset)
        ]
        results = pool.starmap(process_one, args)
        pickle.dump(results, open(os.path.join(output_dir, "results.pkl"), "wb"))
    
    for response, validation_results in results:
        # Save the validation results
        print(validation_results)


def LLM_predict_openai_API_batch(dataset: Dataset, output_dir: str, batch_size: int = 4):
    for idx in range(0, len(dataset), batch_size):
        batch = dataset.select(range(idx, idx+batch_size))
        prompt_list = [prepare_prompt(record) for record in batch]
        response = huoshan_deepseek_r1_batch(client, prompt_list)
        pickle.dump(response, open(os.path.join(output_dir, f"response_{idx}_{idx+batch_size}.pkl"), "wb"))


def validate_all(dataset, output_dir):
    args = [
        (pickle.load(open(os.path.join(output_dir, f"response_{idx}.pkl"), "rb")),
            record, idx, output_dir)
        for idx, record in enumerate(dataset)
    ]
    with Pool(processes=40) as pool:
        results = pool.starmap(evaluate_response, args)
    for r in results:
        print(r["idx"], r["predict_compile_success"], r["predict_execution_success"], r["target_compile_success"], r["target_execution_success"], dataset[r["idx"]]["llvm_ir"]["bb_count"])
    pickle.dump(results, open(os.path.join(output_dir, "validation_results.pkl"), "wb"))
    predict_compile_success_count = sum(1 for result in results if result["predict_compile_success"])
    predict_execution_success_count = sum(1 for result in results if result["predict_execution_success"])
    target_compile_success_count = sum(1 for result in results if result["target_compile_success"])
    target_execution_success_count = sum(1 for result in results if result["target_execution_success"])
    print(f"Number of predict_compile_success: {predict_compile_success_count}")
    print(f"Number of predict_execution_success: {predict_execution_success_count}")
    print(f"Number of target_compile_success: {target_compile_success_count}")
    print(f"Number of target_execution_success: {target_execution_success_count}")
    

def main(dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100",
        #  output_dir = "validation/openai-gpt/gpt-4.1-assembly-with-comments"
        output_dir = "validation/deepseek-r1/deepseek-r1-assembly-with-comments/"
         ):
    
    dataset = load_from_disk(dataset_dir_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # response = pickle.load(open(os.path.join(output_dir, "response_0_4.pkl"), "rb"))
    # print(response.choices[0].message.content)
    # validation_results = evaluate_response(response, dataset[3], 3, output_dir)
    # print(validation_results)
    # LLM_predict_openai_API(dataset, output_dir, num_processes=16, remove_comments=False)
    # LLM_predict_openai_API_batch(dataset, output_dir, batch_size=4)

    validate_all(dataset, output_dir)



def correct_one(chat_response, idx: int, record: dict, output_dir: str, validation_result: dict, num_retry: int = 10) -> bool:
    predict = extract_llvm_code_from_response(chat_response)
    sample_dir = os.path.join(output_dir, f"sample_{idx}")
    count = 0
    predict_compile_success, predict_execution_success = validation_result["predict_compile_success"], validation_result["predict_execution_success"]
    while count < num_retry and (not predict_compile_success or not predict_execution_success):
        count += 1
        logger.info(f"Retrying {count} times for index {idx}")
        if not validation_result["predict_compile_success"]:
            error_msg = ""
            with open(os.path.join(sample_dir, "predict.error"), 'r') as f:
                error_msg = f.read()
            if error_msg.strip() == "":
                logging.warning(f"Error message is empty for index {idx}")
                break
            prompt = format_compile_prompt(record["asm"]["code"][-1], predict, error_msg)
        elif not validation_result["predict_execution_success"]:
            predict_assembly = ""
            with open(os.path.join(sample_dir, "predict.s"), 'r') as f:
                predict_assembly = f.read()
            if predict_assembly.strip() == "":
                logging.warning(f"Assembly code is empty for index {idx}")
                break
            prompt = format_execution_prompt(record["asm"]["code"][-1], predict, predict_assembly)
        response = huoshan_deepseek_r1(client, prompt)
        pickle.dump(response, open(os.path.join(output_dir, f"response_{idx}_retry_{count}.pkl"), "wb"))
        predict = extract_llvm_code_from_response(response)
        predict_compile_success, predict_assembly_path = compile_llvm_ir(predict, sample_dir, name_hint="predict")
        if predict_compile_success:
            with open(predict_assembly_path, 'r') as f:
                predict_execution_success = eval_assembly(record, f.read())

    if predict_execution_success:
        with open(os.path.join(sample_dir, "corrected_llvm_ir.ll"), 'w') as f:
            f.write(predict)
        logger.info(f"Corrected LLVM IR saved to {os.path.join(sample_dir, 'corrected_llvm_ir.ll')}")
        return True
    else:
        logger.warning(f"Failed to correct LLVM IR for index {idx} after {num_retry} retries.")
        return False
        

def fix_all():
    dir_path = "validation/deepseek-r1/deepseek-r1-assembly-with-comments/"
    response = pickle.load(open(os.path.join(dir_path, "results.pkl"), "rb"))
    dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"
    dataset = load_from_disk(dataset_dir_path)
    fix_count = 0
    for idx, (chat, validation) in enumerate(response):
        if not validation["predict_compile_success"] or not validation["predict_execution_success"]:
            print(f"Index: {idx}, {validation['predict_compile_success']}, {validation['predict_execution_success']}")
            success = correct_one(chat, idx, dataset[idx], dir_path, validation)
            if success:
                fix_count += 1
    print("Total fixed: ", fix_count)


if __name__ == "__main__":
    # main()
    fix_all()
