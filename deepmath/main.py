# train_grpo.py
import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from math_verify import parse
from transformers import TrainerCallback
if os.environ.get("ON_NEURON") == "1":
    import torch_neuronx._C as _C

import torch
torch.manual_seed(0)

class NeuronCacheDiagnosticsCallback(TrainerCallback):
    """Logs Neuron cache sizes per training step to diagnose OOM / recompilation."""

    def on_step_end(self, args, state, control, **kwargs):
        stats = _C._get_compilation_cache_stats()
        print(
            f"[NEURON_CACHE] step={state.global_step} "
            f"entries={stats['total_entries']} "
            f"mem_MB={stats['memory_usage_bytes'] / 1e6:.1f} "
            f"hits={stats['cache_hits']} "
            f"misses={stats['cache_misses']} "
            f"hit_rate={stats['hit_rate']:.3f} "
            f"compilations={stats['total_compilations']} "
            f"compile_time_s={stats['total_compilation_time_ms'] / 1000:.1f}"
        )


class PrintRolloutsCallback(TrainerCallback):
    """Prints every prompt/completion pair from the latest rollout on rank 0.

    Reads trainer._logs, which GRPOTrainer populates per generation step when
    args.log_completions is true.
    """

    def __init__(self, trainer):
        self._trainer = trainer
        self._last_printed_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step == self._last_printed_step:
            return
        logs = self._trainer._logs
        prompts = list(logs["prompt"])
        completions = list(logs["completion"])
        if not prompts:
            return
        print(f"\n===== [ROLLOUT] step={state.global_step} n={len(prompts)} =====")
        for i, (p, c) in enumerate(zip(prompts, completions)):
            print(f"--- [{i}] PROMPT ---")
            print(p)
            print(f"--- [{i}] COMPLETION ---")
            print(c if isinstance(c, str) else c[0].get("content", c))
        print("===== [/ROLLOUT] =====\n", flush=True)
        self._last_printed_step = state.global_step

parser = TrlParser(GRPOConfig)
(config,) = parser.parse_args_and_config()

if os.environ.get("SLURM_TB_LOG_DIR"):
    config.logging_dir = os.environ["SLURM_TB_LOG_DIR"]

SYSTEM_PROMPT = "Solve the math problem. Put your final answer in \\boxed{}."

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
dataset = dataset.filter(lambda x: len(parse(x["solution"])) > 0)
dataset = dataset.map(
    lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}] + x["prompt"]}
)

trainer = GRPOTrainer(
    args=config,
    # model="Qwen/Qwen2-0.5B-Instruct",
    model="Qwen/Qwen3-0.6B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    callbacks=[NeuronCacheDiagnosticsCallback()] if os.environ.get("ON_NEURON") == "1" else [],
)
trainer.add_callback(PrintRolloutsCallback(trainer))

# Accelerate rollout on Neuron by compiling model.forward with the neuron backend.
# Mirrors TorchNeuronEager/examples/torch_compile/qwen3_0_6b/run_qwen3_0_6b.py.
# Requires static KV cache (set via `cache_implementation: static` in the yaml)
# so decode-step shapes stay fixed and the graph is reused.
if os.environ.get("ON_NEURON") == "1" and not config.use_vllm and not config.use_nkipy:
    # Compile only the rollout forward. _fsdp2_unshard_for_generation swaps
    # model.forward to model._rollout_forward for the duration of rollout,
    # where FSDP2 hooks and activation checkpointing are suspended so
    # fullgraph=True is safe. Training keeps the eager forward (which still
    # goes through the FSDP2 hooks for sharded compute).
    trainer.model._rollout_forward = torch.compile(
        trainer.model.forward, backend="neuron", fullgraph=True, dynamic=False
    )

trainer.train()
