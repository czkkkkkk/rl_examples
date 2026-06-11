
# export SLURM_TB_LOG_DIR=output/runs/tb-${SLURM_JOB_ID}

export TORCH_NEURONX_NEFF_CACHE_DIR="/fsx/zkcai/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/fsx/zkcai/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
# FSDP=16 training pins cores 0-15; the DP=16 rollout server pins cores 16-31.
# Launch hf_serve/run_hf_serve_qwen8b_dp16tp1.sh in a separate shell FIRST.
export NEURON_RT_VISIBLE_CORES=0-15

# export XLA_IR_DEBUG=1
# export XLA_HLO_DEBUG=1
export DISABLE_VLLM_WEIGHT_SYNC=0
export TRL_PROFILE_STDOUT=1

# Disable CPU fallback: raise instead of silently offloading ops to CPU.
# Training runs the model in eager mode (no torch.compile), so unsupported ops
# are routed through torch_neuronx's eager dispatcher, which by default copies
# tensors to CPU, runs the op there, and copies back. That hides which ops lack
# a Neuron implementation (notably some backward ops). With the flags below, any
# op that can't execute on Neuron raises a RuntimeError naming the op instead.
#   - FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1: main eager-dispatch path (Python
#     OperationImplementation + C++ NeuronDispatcher). Raises for both the
#     "Neuron impl failed" and "no impl could handle" cases.
#   - CONTIGUOUS_FALLBACK_FATAL=1: the separate contiguous/stride CPU-fallback
#     path is also made fatal rather than silently offloaded.
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
export TORCH_NEURONX_CONTIGUOUS_FALLBACK_FATAL=1

# Train Qwen3-8B (overrides main.py's default 0.6B).
export GRPO_MODEL="Qwen/Qwen3-8B"

# Pre-downloaded model: force offline so the 16 FSDP ranks don't race the Hub
# cache and transiently see a missing shard.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# RAM-efficient FSDP load: only LOCAL_RANK 0 reads the checkpoint; other ranks
# instantiate empty tensors and receive shards via FSDP sync_module_states.
# transformers reads these from the ENV (not the accelerate yaml), so export here.
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

source /fsx/zkcai/miniconda3/bin/activate test_eager
SCRIPT_DIR=$(dirname "$(realpath "$0")")

CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/fsdp16.yaml"
GRPO_CONFIG="$SCRIPT_DIR/grpo_configs/grpo_hf_serve_qwen8b.yaml"
accelerate launch \
    --config_file "$CONFIG_FILE" \
    "$SCRIPT_DIR/main.py" \
    --config "$GRPO_CONFIG"
