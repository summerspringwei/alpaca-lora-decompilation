docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:`pwd`/data \
    ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id 