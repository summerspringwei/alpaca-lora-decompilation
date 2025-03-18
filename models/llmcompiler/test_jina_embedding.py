from transformers import AutoModel
from numpy.linalg import norm
import json

from utils.preprocessing_llvm_ir import  preprocessing_llvm_ir

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
embeddings = model.encode(
    [
        'How do I access the index while iterating over a sequence with a for loop?',
        '# Use the built-in enumerator\nfor idx, x in enumerate(xs):\n    print(idx, x)',
    ]
)
print(cos_sim(embeddings[0], embeddings[1]))

validation_file_path = "/home/xiachunwei/Projects/alpaca-lora-decompilation/validation/exebench_llmcompiler/exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-length_reward_pack_similar_length_samples-step-40-bs-32-beams-1_validate_exebench.json"

records = json.load(open(validation_file_path, 'r'))
start, batch = 0, 16
for start in range(0, len(records), batch):
    group = records[start:start+batch]
    for g in group:
        g["output"] = preprocessing_llvm_ir(g["output"])
    predict_embedding = model.encode([g["predict"][0] for g in group])
    output_embedding = model.encode([g["output"] for g in group])
    
    for g, pe, oe in zip(group, predict_embedding, output_embedding):
        print(g["predict_compile_success"], g["predict_execution_success"], cos_sim(pe, oe))
        print("<<<"*20)
        print(g["predict"][0])
        print("==="*20)
        print(g["output"])
        print(">>>"*20)
        # if not g["predict_compile_success"][0]:
        #     print(g["predict"][0])
        #     print("==="*20)
        #     print(g["output"])

    # exit(0)

