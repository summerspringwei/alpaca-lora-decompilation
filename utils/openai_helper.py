import re
import logging

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


def extract_llvm_code_from_response(response):
    if response.choices and len(response.choices) > 0:
        result = response.choices[0].message.content
        # extract the content after </think>
        if result.find("</think>") != -1:
            result = result.split("</think>")[-1].strip()
        llvm_code = extract_llvm_code(result)
        if len(llvm_code) == 0:
            logger.warning(f"No LLVM code found in the response: {result}")
        return llvm_code[0] if len(llvm_code) > 0 else ""
    else:
        logger.warning("No choices found in the response.")
    return ""


def format_compile_prompt(target_assembly, predict, error_msg):
    prompt = f"""Please decompile the following assembly code to LLVM IR and please place the final generated LLVM IR code between ```llvm and ```:\n
            {target_assembly} \n 
            Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once.\n
            You generated the following LLVM IR but it is failed to be compiled: ```llvm\n{predict}```\n 
            The compilation error message is as follows: {error_msg}
            Please correct the LLVM IR code and make sure it is correct.\n
            place the final generated LLVM IR code between ```llvm and ```.
            """
    return prompt


def format_execution_prompt(target_assembly, predict, predict_assembly):
    prompt = f"""Please decompile the following assembly code to LLVM IR and please place the final generated LLVM IR code between ```llvm and ```:\n
        {target_assembly} \n 
        Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once.\n
        Then you generated the following LLVM IR: ```llvm\n{predict}```\n 
        After I compile the LLVM IR you provided, the generated assembly is: {predict_assembly}\n
        The result is not right. Please compare with the original result and re-generate the LLVM IR.\n
        Place the final generated LLVM IR code between ```llvm and ```.
        """

    return prompt
