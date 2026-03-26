# vllm serve Qwen/Qwen3.5-35B-A3B --port 8010 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
# vllm serve Qwen/Qwen3.5-27B --port 8010 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only




# are-benchmark run \
#     --hf-dataset datasets/gaia2/ \
#     --hf-split validation \
#     --hf-config mini \
#     --agent default \
#     --model hosted_vllm/Qwen/Qwen3.5-27B \
#     --provider local \
#     --endpoint "http://localhost:8010/v1" \
#     --output_dir ./results/validation \
#     -l 1



are-benchmark run \
    --hf-dataset datasets/gaia2/ \
    --hf-split validation \
    --hf-config mini \
    --agent default \
    --model hosted_vllm/Qwen/Qwen3.5-35B-A3B \
    --provider local \
    --endpoint "http://localhost:8010/v1" \
    --output_dir ./results/validation-2 \
    --judge_model hosted_vllm/Qwen/Qwen3.5-35B-A3B \
    --judge_provider local \
    --judge_endpoint "http://localhost:8010/v1" \
    --a2a_app_prop 1 \
    --a2a_model hosted_vllm/Qwen/Qwen3.5-35B-A3B \
    --a2a_model_provider local \
    --a2a_endpoint "http://localhost:8010/v1" \
    -l 3
 