
import json
import re
import utils

from utils import preprocessing_assembly

def extract_llmcompiler_code_blocks(text):
    pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    matches = pattern.findall(text)
    return matches

datas = json.load(open("ppo-llm-compiler-bs-32-mbs-8.json", 'r'))

for record in datas:
    print('-'*20)
    out = extract_llmcompiler_code_blocks(record['predict'])
    print(record['input'])
    if len(out) > 0:
        print(out[0])
    print('-'*20)


a = '.ident  "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bf774c565080e30e0a073676c316ab175303af)"'
b = '.file   "exebench_lscat-ACT41_2318108cwm4j9m1.c"'
from transformers import AutoTokenizer
file_path = "facebook/llm-compiler-7b-ftd"
tokenizer = AutoTokenizer.from_pretrained(file_path)

print(len(tokenizer(a)['input_ids']))
print(len(tokenizer(b)['input_ids']))
