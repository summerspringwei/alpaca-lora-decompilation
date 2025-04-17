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
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.preprocessing_assembly import preprocessing_assembly

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_llvm_code(markdown_content: str):
    llvm_code_blocks = []
    # Use a non-greedy regex to match multiple code blocks
    pattern = r"```llvm\n(.*?)\n```"  # The \n is crucial to prevent matching across blocks
    matches = re.findall(pattern, markdown_content, re.DOTALL) # re.DOTALL to match across multiple lines

    if matches:
        llvm_code_blocks = matches

    return llvm_code_blocks


# def gemini_predict(dataset_path: str, output_path: str):

#     api_key = os.getenv('GEMINI_API_KEY').strip()
#     client = genai.Client(api_key=api_key, http_options={'api_version':'v1alpha'})
#     dataset = load_from_disk(dataset_path)

#     try:
#         with open(output_path, 'a') as f:
#             for p in dataset:
#                 asm_code = p["asm"]["code"][-1]
#                 input_str = "decompile the x86 assembly to llvm ir: \n" + asm_code

#                 response = client.models.generate_content(
#                     model='gemini-2.0-flash-thinking-exp', contents=input_str, config={
#                         "response_logprobs": True, "response_lengt": 10
#                     }
#                 )
#                 parts = [part.text for part in response.candidates[0].content.parts]
#                 # We need to extract the LLVM code from the response
#                 llvm_code = [extract_llvm_code(part) for part in parts]
#                 llvm_code = [code[0] for code in llvm_code if len(code) > 0]
#                 predict = llvm_code[0] if len(llvm_code) > 0 else ""
#                 out = {
#                                 "instruction": input_str,
#                                 "input": asm_code,
#                                 "predict": predict,
#                                 "raw_response": parts,
#                                 "file": p["path"],
#                                 "output": p["llvm_ir"]["code"],
#                                 "func_head_types": p["func_head_types"]
#                 }
#                 json.dump(out, f)
#                 f.write('\n')
#                 f.flush()
#     except Exception as e:
#         print(f"An error occurred: {e}")

global client
client = OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3",timeout=1800)


def huoshan_deepseek_r1(client, prompt: str):
    response = client.chat.completions.create(
                model='ep-20250317013717-m9ksl',
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


def extract_llvm_code_from_response(response):
    if response.choices and len(response.choices) > 0:
        result = response.choices[0].message.content
        llvm_code = extract_llvm_code(result)
        if len(llvm_code) == 0:
            logger.warning(f"No LLVM code found in the response: {result}")
        return llvm_code[0] if len(llvm_code) > 0 else ""
    else:
        logger.warning("No choices found in the response.")
    return ""


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
    validation_results = evaluate_response(response, record, idx, output_dir)
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
    predict_compile_success_count = sum(1 for result in results if result["predict_compile_success"])
    predict_execution_success_count = sum(1 for result in results if result["predict_execution_success"])
    target_compile_success_count = sum(1 for result in results if result["target_compile_success"])
    target_execution_success_count = sum(1 for result in results if result["target_execution_success"])
    print(f"Number of predict_compile_success: {predict_compile_success_count}")
    print(f"Number of predict_execution_success: {predict_execution_success_count}")
    print(f"Number of target_compile_success: {target_compile_success_count}")
    print(f"Number of target_execution_success: {target_execution_success_count}")
    

def main():
    dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"
    dataset = load_from_disk(dataset_dir_path)
    output_dir = os.path.join("validation/deepseek-r1", "deepseek-r1-assembly-with-comments")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # response = pickle.load(open(os.path.join(output_dir, "response_0_4.pkl"), "rb"))
    # print(response.choices[0].message.content)
    # validation_results = evaluate_response(response, dataset[3], 3, output_dir)
    # print(validation_results)
    # LLM_predict_openai_API(dataset, output_dir, num_processes=100, remove_comments=False)
    # LLM_predict_openai_API_batch(dataset, output_dir, batch_size=4)

    validate_all(dataset, output_dir)


if __name__ == "__main__":
    main()
