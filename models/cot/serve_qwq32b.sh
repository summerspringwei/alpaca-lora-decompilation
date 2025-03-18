ssh -N -R 8.141.164.33:8080:localhost:9001 root@8.141.164.33

export CUDA_VISIBLE_DEVICES=2,3 &&
vllm serve ~/Datasets/Models/QwQ-32B-Preview/ --dtype auto --port 9001 --max-model-len 15000 --api-key token-llm4decompilation-abc123 --pipeline-parallel-size 2