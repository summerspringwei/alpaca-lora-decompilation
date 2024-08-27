import json

# "predict_compile_success":true,
#         "predict_execution_success":true,
#         "target_compile_success":true,
#         "target_execution_success":true

records = json.load(open("exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd_validate_exebench.json", 'r'))
predict_compile = [r["predict_compile_success"] for r in records]
predict_execution = [r["predict_execution_success"] for r in records]
target_compile = [r["target_compile_success"] for r in records]
target_execution = [r["target_execution_success"] for r in records]

print(sum(predict_compile), sum(predict_execution), sum(target_compile), sum(target_execution))
