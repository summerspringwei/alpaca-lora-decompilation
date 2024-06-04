from dataclasses import dataclass
from typing import Optional, Any, Union
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import torch
from torch import nn
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from transformers.utils.generic import PaddingStrategy

from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from safetensors import safe_open

encoder_max_length = 1024*7
decoder_max_length = 1024*9


pretrained_model_path = "/data/xiachunwei/Datasets/Models/bart-large"
finetuned_model_path = "/data/xiachunwei/Projects/alpaca-lora-decompilation/bart-large-decompilation/checkpoint-12500/model.safetensors"


def get_bart_model(pretrained_model_path: str, finetuned_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    led = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path)
    led.base_model.encoder.embed_positions = BartLearnedPositionalEmbedding(encoder_max_length, 1024)
    nn.init.xavier_uniform_(led.base_model.encoder.embed_positions.weight)
    led.base_model.decoder.embed_positions = BartLearnedPositionalEmbedding(decoder_max_length, 1024)
    nn.init.xavier_uniform_(led.base_model.decoder.embed_positions.weight)

    led.config.num_beams = 4
    led.config.max_length = encoder_max_length
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    state_dict = {}
    with safe_open(finetuned_model_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    led.load_state_dict(state_dict)
    led.to("cuda")
    led.eval()
    return led, tokenizer


def evaluate(model, tokenizer, input:str):
    inputs = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=encoder_max_length)
    input_ids = inputs["input_ids"].to("cuda")
    temperature=0.1
    top_p=0.75
    top_k=50
    generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            max_new_tokens=1024*9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.config.bos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=False
        )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids = inputs["input_ids"].to("cuda"),
            attention_mask = inputs["attention_mask"].to("cuda"),
            generation_config = generation_config
            # num_beams=2, min_length=0, max_length=decoder_max_length
        )
        import pickle
        pickle.dump(generation_output, open("generation_output.pkl", "wb"))
    outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # s = generation_output.sequences
    # outputs = tokenizer.decode(s, skip_special_tokens=True)
    return outputs


def my_generate(model, tokenizer, input:str):
    inputs = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=encoder_max_length)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    count = 0
    with torch.no_grad():
        decoder_input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to("cuda")
        str = ""
        while True:
            output = model.forward(input_ids, attention_mask, decoder_input_ids=decoder_input_ids, return_dict=True)
            logits = output.logits
            logits = logits[:, -1, :]
            idx = torch.argmax(logits, dim=-1)
            decoder_input_ids = torch.cat((decoder_input_ids, idx.unsqueeze(0)), dim=-1)
            count+=1
            if count > 300:
                break
            tok = tokenizer.decode(idx)
            print(tok)
            str += tok
        print(str)

def test():
    model, tokenizer = get_bart_model(pretrained_model_path, finetuned_model_path)
    input_str = '\t.text\n\t.file\t"extr_Server.c_IncrementServerConfigRevision.ll"\n\t.globl\tIncrementServerConfigRevision   # -- Begin function IncrementServerConfigRevision\n\t.p2align\t4, 0x90\n\t.type\tIncrementServerConfigRevision,@function\nIncrementServerConfigRevision:          # @IncrementServerConfigRevision\n\t.cfi_startproc\n# %bb.0:\n\ttestq\t%rdi, %rdi\n\tje\t.LBB0_2\n# %bb.1:\n\tincl\t(%rdi)\n.LBB0_2:\n\tretq\n.Lfunc_end0:\n\t.size\tIncrementServerConfigRevision, .Lfunc_end0-IncrementServerConfigRevision\n\t.cfi_endproc\n                                        # -- End function\n\t.ident\t"Ubuntu clang version 15.0.7"\n\t.section\t".note.GNU-stack","",@progbits\n'
    output_str = 'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\ntarget triple = "x86_64-pc-linux-gnu"\n\n; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable\ndefine dso_local void @IncrementServerConfigRevision(ptr noundef %0) local_unnamed_addr #0 {\n  %2 = icmp eq ptr %0, null\n  br i1 %2, label %6, label %3\n\n3:                                                ; preds = %1\n  %4 = load i32, ptr %0, align 4, !tbaa !5\n  %5 = add nsw i32 %4, 1\n  store i32 %5, ptr %0, align 4, !tbaa !5\n  br label %6\n\n6:                                                ; preds = %1, %3\n  ret void\n}\n\nattributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }\n\n!llvm.module.flags = !{!0, !1, !2, !3}\n!llvm.ident = !{!4}\n\n!0 = !{i32 1, !"wchar_size", i32 4}\n!1 = !{i32 7, !"PIC Level", i32 2}\n!2 = !{i32 7, !"PIE Level", i32 2}\n!3 = !{i32 7, !"uwtable", i32 2}\n!4 = !{!"Ubuntu clang version 15.0.7"}\n!5 = !{!6, !7, i64 0}\n!6 = !{!"TYPE_3__", !7, i64 0}\n!7 = !{!"int", !8, i64 0}\n!8 = !{!"omnipotent char", !9, i64 0}\n!9 = !{!"Simple C/C++ TBAA"}\n'
    output = evaluate(model, tokenizer, input_str)
    print(output)
    # my_generate(model, tokenizer, input_str)
    print("*="*20)
    print(output_str)
    


if __name__ == "__main__":
    test()
