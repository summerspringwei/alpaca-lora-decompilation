from datasets import load_dataset, load_metric, load_from_disk
print('target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\ntarget triple = "x86_64-pc-linux-gnu"\n\n; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable\ndefine dso_local void @IncrementServerConfigRevision(ptr noundef %0) local_unnamed_addr #0 {\n  %2 = icmp eq ptr %0, null\n  br i1 %2, label %6, label %3\n\n3:                                                ; preds = %1\n  %4 = load i32, ptr %0, align 4, !tbaa !5\n  %5 = add nsw i32 %4, 1\n  store i32 %5, ptr %0, align 4, !tbaa !5\n  br label %6\n\n6:                                                ; preds = %1, %3\n  ret void\n}\n\nattributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }\n\n!llvm.module.flags = !{!0, !1, !2, !3}\n!llvm.ident = !{!4}\n\n!0 = !{i32 1, !"wchar_size", i32 4}\n!1 = !{i32 7, !"PIC Level", i32 2}\n!2 = !{i32 7, !"PIE Level", i32 2}\n!3 = !{i32 7, !"uwtable", i32 2}\n!4 = !{!"Ubuntu clang version 15.0.7"}\n!5 = !{!6, !7, i64 0}\n!6 = !{!"TYPE_3__", !7, i64 0}\n!7 = !{!"int", !8, i64 0}\n!8 = !{!"omnipotent char", !9, i64 0}\n!9 = !{!"Simple C/C++ TBAA"}\n')
exit(0)
dataset_train = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort")
dataset_val = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort")

for i in range(100):
    print(dataset_train[-i]['output'])
    
    print("\n"*3)



# import os
# import torch
# from safetensors import safe_open
# model_path = "/data/xiachunwei/Projects/alpaca-lora-decompilation/bart-large-decompilation/checkpoint-12500"
# file_path = os.path.join(model_path, "model.safetensors")
# f = safe_open(file_path, framework="pt", device="cuda")
# state_dict = {}
# for k in f.keys():
#     state_dict[k] = f.get_tensor(k)
#     print(k)
# print(f.get_tensor("model.encoder.embed_positions.weight").shape)
# print(f.get_tensor("model.decoder.embed_positions.weight").shape)


# file_path = "/data/xiachunwei/Projects/alpaca-lora-decompilation/bart-large-decompilation/checkpoint-12500/global_step12500/mp_rank_00_model_states.pt"
# f = torch.load(open(file_path, "rb"))

# print(f["module"]["model.decoder.layernorm_embedding.weight"].shape)


from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("/data/xiachunwei/Datasets/Models/bart-large")
tokenizer = AutoTokenizer.from_pretrained("/data/xiachunwei/Datasets/Models/bart-large")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]