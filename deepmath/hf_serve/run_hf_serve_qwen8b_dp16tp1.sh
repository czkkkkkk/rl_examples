#!/usr/bin/env bash
# Qwen/Qwen3-8B perf-benchmark server: tp_size=8, dp_size=1 (world_size=8),
# prompt padded to 256, compiled Neuron path.
#
#   prompt shape : max_prompt_length=256, pad_to_multiple=256  -> every prompt
#                  left-padded to exactly 256 tokens (one fixed prefill shape).
#   generation   : max_new_tokens set per request (benchmark uses 1024), so the
#                  static KV cache is sized 256 + 1024 = 1280.
#
# Compiled by default via the env gates below (fullgraph=False is required for
# tp>1). Cores 8-15, one per rank, clear of training cores 0-7.

export TORCH_NEURONX_NEFF_CACHE_DIR="/fsx/zkcai/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/fsx/zkcai/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
export NEURON_RT_VISIBLE_CORES=16-31
# DP path: tp_size==1, compiled automatically (no HF_SERVE_COMPILE_TP)
export HF_SERVE_COMPILE_FULLGRAPH=1
# Quiet the per-rollout decode/print so it doesn't skew timings.
export HF_SERVE_PRINT_ROLLOUTS=0

source /fsx/zkcai/miniconda3/bin/activate test_eager

trl hf-serve \
    --model Qwen/Qwen3-8B \
    --host 127.0.0.1 \
    --port 30000 \
    --zmq-port 5558 \
    --dtype bfloat16 \
    --attn-implementation eager \
    --device neuron \
    --max-prompt-length 256 \
    --pad-to-multiple 256 \
    --cache-implementation static \
    --world-size 16
