#!/usr/bin/env bash
# Launch the HF serve-mode rollout server on a dedicated set of Neuron cores.
# Run this in its own shell before starting ../run_neuron.sh on cores 0-7.

export TORCH_NEURONX_NEFF_CACHE_DIR="/fsx/zkcai/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/fsx/zkcai/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
export NEURON_RT_VISIBLE_CORES=8-11

# Compile the tensor-parallel forward into a single NEFF. The server strips the
# TP backward hooks before compiling, so torch.compile(fullgraph=True) traces
# the whole forward without gb0083 ("Module-level backwards hooks require
# compiled autograd"). HF_SERVE_COMPILE_TP=1 opts the tp_size>1 path into
# torch.compile (off by default); fullgraph stays on.
export HF_SERVE_COMPILE_TP=1
export HF_SERVE_COMPILE_FULLGRAPH=1

source /fsx/zkcai/miniconda3/bin/activate eager_v3

trl hf-serve \
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
    --world-size 4 \
    --tp-size 2
