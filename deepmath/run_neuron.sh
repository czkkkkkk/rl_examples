
# export SLURM_TB_LOG_DIR=output/runs/tb-${SLURM_JOB_ID}

export TORCH_NEURONX_NEFF_CACHE_DIR="/fsx/zkcai/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/fsx/zkcai/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
# Training pins cores 0-7; run ./hf_serve/run_hf_serve.sh in a separate
# shell first — it pins cores 8-15 for the rollout server.
export NEURON_RT_VISIBLE_CORES=0-7
# export ACCELERATE_USE_FSDP=true

# export XLA_IR_DEBUG=1
# export XLA_HLO_DEBUG=1
export DISABLE_VLLM_WEIGHT_SYNC=0

source /fsx/zkcai/miniconda3_v2/bin/activate eager
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/ddp.yaml"
CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/fsdp.yaml"
GRPO_CONFIG="$SCRIPT_DIR/grpo_configs/grpo_hf_serve.yaml"
# GRPO_CONFIG="$SCRIPT_DIR/grpo_configs/grpo_hf_generate.yaml"
# GRPO_CONFIG="$SCRIPT_DIR/grpo_configs/grpo_nkipy.yaml"
# GRPO_CONFIG="$SCRIPT_DIR/grpo_configs/grpo.yaml"
accelerate launch \
    --config_file "$CONFIG_FILE" \
    "$SCRIPT_DIR/main.py" \
    --config "$GRPO_CONFIG"
