import os
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch import nn
import fire


def get_peft_state_dict(peft_weight_dir):
    # Load state dict from peft
    checkpoint_name = os.path.join(peft_weight_dir, "adapter_model.safetensors")
    from safetensors import safe_open
    f = safe_open(checkpoint_name, framework="pt", device="cuda")
    adapters_weights = {}
    for k in f.keys():
        print(k)
        adapters_weights[k] = f.get_tensor(k)
    
    peft_state_dict = {}
    for k, v in adapters_weights.items():
        str_to_remove = "base_model.model."
        if str_to_remove in k:
            k = k.replace(str_to_remove, "")
        peft_state_dict[k] = v
    
    return peft_state_dict


class ComposeLoRAWeight(nn.Module):
    base_model_weight: torch.Tensor = None
    lora_a: torch.Tensor = None
    lora_b: torch.Tensor = None

    def forward(self):
        lora_a = self.lora_a.to(torch.float32)
        lora_b = self.lora_b.to(torch.float32)
        base_model_weight = self.base_model_weight.to(torch.float32)
        print(base_model_weight.shape)
        print(lora_a.shape)
        print(lora_b.shape)
        return (base_model_weight + (lora_b @ lora_a)).to(torch.float16)

    def isNone(self):
        return self.lora_a is None or self.lora_b is None


def main(base_model="/data0/xiachunwei/Dataset/CodeLlama-7b-hf",
    merged_model = "/data0/xiachunwei/Dataset/CodeLlama-7b-hf_merged_lora",
    peft_weight_dir = "./decompile_llvm_ir_alphaca_lora_seq_len_4k_with_flashattn_maybe_wrong/checkpoint-4100"
    ):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    model.to("cuda")
    base_model_state_dict = model.state_dict()

    # Name of base model state dict: model.layers.0.self_attn.q_proj.weight
    # Name of peft weights: base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight

    peft_state_dict = get_peft_state_dict(peft_weight_dir)

    for name, value in base_model_state_dict.items():
        striped_name = name.replace(".weight", "")
        compose = ComposeLoRAWeight()
        compose.base_model_weight = value
        for k, v in peft_state_dict.items():
            if striped_name in k:
                if "lora_A" in k:
                    compose.lora_a = v
                elif "lora_B" in k:
                    compose.lora_b = v
        if not compose.isNone():
            print(name)
            base_model_state_dict[name] = compose.forward()


    model.load_state_dict(base_model_state_dict, strict=True)
    model.save_pretrained(merged_model)
    tokenizer.save_pretrained(merged_model)


if __name__=="__main__":
    fire.Fire(main)
