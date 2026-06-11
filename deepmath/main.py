# train_grpo.py
import os
import time
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from math_verify import parse
from transformers import TrainerCallback
if os.environ.get("ON_NEURON") == "1":
    import torch_neuronx
    import torch_neuronx._C as _C

import torch
torch.manual_seed(0)


def _patch_accelerate_gather_for_neuron():
    """Flatten tensors before all_gather_into_tensor in accelerate's _gpu_gather.

    torch_neuronx's all_gather lowering indexes tensor.shape[0] of the input;
    a 0-d/scalar input (e.g. per-rank reward tensors) raises
    `IndexError: tuple index out of range`. The output buffer is already sized
    by numel, so gathering a flat view is equivalent.
    """
    from accelerate.utils import operations

    def _gpu_gather(tensor):
        state = operations.PartialState()

        def _gpu_gather_one(t):
            if t.ndim == 0:
                t = t.clone()[None]
            if not t.is_contiguous():
                t = t.contiguous()
            output = torch.empty(
                state.num_processes * t.numel(), dtype=t.dtype, device=state.device
            )
            torch.distributed.all_gather_into_tensor(output, t.view(-1))
            return output.view(-1, *t.size()[1:])

        return operations.recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)

    # accelerate's public `gather` dispatches to operations._gpu_gather at call
    # time, so patching the module attribute is enough for trl's imports too.
    operations._gpu_gather = _gpu_gather


if os.environ.get("ON_NEURON") == "1":
    _patch_accelerate_gather_for_neuron()


def _neuron_sync():
    """Flush async Neuron kernels so perf_counter measures real device time."""
    try:
        import torch_neuronx
        torch_neuronx.synchronize()
    except Exception:
        pass


def _neuron_mem_log(label, step):
    """Print per-rank Neuron device memory (current + in-step peak) at a phase
    boundary, then reset the peak so each phase's high-water mark is isolated.
    GRPOTrainer.training_step / _prepare_inputs call this via trainer.memory_profiler
    at before_rollout, after_rollout, after_forward, after_backward; the optimizer
    callback adds after_optimizer_step."""
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    _neuron_sync()  # flush async kernels so the reading reflects real device state
    alloc = torch_neuronx.memory_allocated() / 1e9
    reserved = torch_neuronx.memory_reserved() / 1e9
    peak_alloc = torch_neuronx.max_memory_allocated() / 1e9
    print(
        f"[NEURON_MEM] step={step} rank={rank} phase={label} "
        f"alloc_GB={alloc:.3f} reserved_GB={reserved:.3f} peak_alloc_GB={peak_alloc:.3f}",
        flush=True,
    )
    torch_neuronx.reset_peak_memory_stats()


class NeuronMemoryProfiler:
    """Duck-typed to GRPOTrainer's `memory_profiler` hook: the trainer calls
    `.checkpoint(label, step=...)` at after_rollout / after_forward / after_backward
    (and before_rollout via the trl patch). Each call logs device memory."""

    def checkpoint(self, label, step=None):
        if os.environ.get("TRL_PROFILE_STDOUT") == "1":
            _neuron_mem_log(label, step)


class OptimizerStepTimerCallback(TrainerCallback):
    """Times optimizer.step() in isolation. HF Trainer fires on_pre_optimizer_step
    immediately before optimizer.step() and on_optimizer_step immediately after,
    so the span between them (with Neuron syncs) is the pure optimizer time —
    separate from the backward time measured inside GRPOTrainer.training_step.
    Also emits the after_optimizer_step memory checkpoint (the trainer's
    memory_profiler hooks don't cover optimizer.step, which runs in HF's loop)."""

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if os.environ.get("TRL_PROFILE_STDOUT") == "1":
            _neuron_sync()
            self._t0 = time.perf_counter()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if os.environ.get("TRL_PROFILE_STDOUT") == "1" and hasattr(self, "_t0"):
            _neuron_sync()
            print(
                f"[trl-profile] step={state.global_step} optimizer.step: "
                f"{time.perf_counter() - self._t0:.2f}s",
                flush=True,
            )
            _neuron_mem_log("after_optimizer_step", state.global_step)

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
    # Overridable via GRPO_MODEL so run_neuron.sh can switch to Qwen/Qwen3-8B
    # without editing this file.
    model=os.environ.get("GRPO_MODEL", "Qwen/Qwen3-0.6B"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    callbacks=(
        [NeuronCacheDiagnosticsCallback(), OptimizerStepTimerCallback()]
        if os.environ.get("ON_NEURON") == "1"
        else []
    ),
)
# Attach the device-memory profiler GRPOTrainer calls at phase boundaries
# (before_rollout / after_rollout / after_forward / after_backward). The
# optimizer-step memory point is emitted by OptimizerStepTimerCallback.
if os.environ.get("ON_NEURON") == "1":
    trainer.memory_profiler = NeuronMemoryProfiler()
# Accelerate rollout on Neuron by compiling model.forward with the neuron backend.
# Mirrors TorchNeuronEager/examples/torch_compile/qwen3_0_6b/run_qwen3_0_6b.py.
# Requires static KV cache (set via `cache_implementation: static` in the yaml)
# so decode-step shapes stay fixed and the graph is reused.
if (
    os.environ.get("ON_NEURON") == "1"
    and not config.use_vllm
    and not getattr(config, "use_hf", False)
):
    # Compile only the rollout forward. _fsdp2_unshard_for_generation swaps
    # model.forward to model._rollout_forward for the duration of rollout,
    # where FSDP2 hooks and activation checkpointing are suspended so
    # fullgraph=True is safe. Training keeps the eager forward (which still
    # goes through the FSDP2 hooks for sharded compute).
    trainer.model._rollout_forward = torch.compile(
        trainer.model.forward, backend="neuron", fullgraph=True, dynamic=False
    )

trainer.train()
