#!/usr/bin/env bash
# Launch the HF serve-mode rollout server on a dedicated set of Neuron cores.
# Run this in its own shell before starting ../run_neuron.sh on cores 0-7.

export TORCH_NEURONX_NEFF_CACHE_DIR="/fsx/zkcai/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/fsx/zkcai/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
export NEURON_RT_VISIBLE_CORES=8-11
export HF_SERVE_CPU_MULTINOMIAL=0

source /fsx/zkcai/miniconda3_v2/bin/activate eager
SCRIPT_DIR=$(dirname "$(realpath "$0")")

python "$SCRIPT_DIR/hf_serve.py" \
    --model Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 30000 \
    --zmq-port 5558 \
    --dtype bfloat16 \
    --attn-implementation eager \
    --device neuron \
    --max-prompt-length 512 \
    --pad-to-multiple 128 \
    --cache-implementation static \
    --world-size 4
