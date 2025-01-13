from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils.prompter import Prompter


pretrained_model_path = 
batch_size = 1
enable_lora = True
llm = LLM(model=pretrained_model_path, max_num_seqs=batch_size, enable_lora=enable_lora)
temperature=0.1
top_p=0.95
top_k=10
num_beams=1
max_new_tokens=1024 * 4
sampling_params = SamplingParams(top_k=top_k,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_new_tokens,
                                    min_tokens=256)


prompter = Prompter("alpaca")
promp = prompter.generate_prompt(instruction="", input="")
sequences = llm.generate(promp, sampling_params, lora_request=LoRARequest("lora", 1, lora_adapter_path))
