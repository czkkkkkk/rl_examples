# source /fsx/zkcai/miniconda3/bin/activate grpo_baseline
source /fsx/zkcai/miniconda3_v2/bin/activate eager

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PORT=8001

model_id="Qwen/Qwen2-0.5B-Instruct"
CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
    --model "$model_id" \
    --dtype float32 \
    --vllm_model_impl transformers \
    --port "$PORT"