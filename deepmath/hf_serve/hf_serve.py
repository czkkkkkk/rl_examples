"""Data-parallel HF serve-mode rollout server for GRPO training.

Architecture
------------
Rank 0 (the main process) runs FastAPI + ZMQ. It spawns ``--world-size N``
worker processes; each worker holds a full ``AutoModelForCausalLM`` on its
own slice of Neuron cores and runs ``model.generate()`` under
``torch.compile(backend="neuron")``.

- ``POST /generate`` shards prompts round-robin across workers, collects their
  results, and reassembles by global prompt index.
- ``POST /init_communicator`` + ZMQ REP loop receives each parameter, broadcasts
  it to every worker over ``torch.multiprocessing`` queues, and acks the client
  once every rank has applied the update in place (``param.data.copy_``).

Fixed-shape contract (for torch.compile on Neuron)
--------------------------------------------------
Each worker tokenizes with ``padding="max_length"``, ``max_length =
--max-prompt-length``, so the prefill graph compiled on the first request is
reusable for every subsequent batch. Weight updates go in place, preserving
the compiled graph's captured tensor pointers.

Neuron core slicing
-------------------
The launcher reads ``NEURON_RT_VISIBLE_CORES`` (e.g. ``8-15``), splits it
evenly across ``--world-size`` workers, and sets the per-rank
``NEURON_RT_VISIBLE_CORES`` in each child BEFORE the Neuron runtime
initializes. Example: ``NEURON_RT_VISIBLE_CORES=8-15`` with
``--world-size 4`` gives ranks cores ``8,9``, ``10,11``, ``12,13``,
``14,15``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import threading
import time
from queue import Empty
from typing import Any

import msgpack
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import uvicorn
import zmq
from fastapi import FastAPI
from multiprocessing import shared_memory
from pydantic import BaseModel


logger = logging.getLogger("hf_serve")
logging.basicConfig(level=logging.INFO, format="[HF-Serve] %(asctime)s %(message)s")


_TORCH_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
}


def _parse_torch_dtype(name: str) -> torch.dtype:
    if name in _TORCH_DTYPE_MAP:
        return _TORCH_DTYPE_MAP[name]
    raise ValueError(f"Unsupported torch dtype string from client: {name!r}")


def _parse_core_range(s: str) -> list[int]:
    """Parse ``"a-b"`` / ``"a,b,c-d"`` into an int list."""
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _slice_cores(cores: list[int], world_size: int) -> list[str]:
    """Split ``cores`` into ``world_size`` contiguous chunks (strings)."""
    if not cores:
        return ["" for _ in range(world_size)]
    n = len(cores)
    if n < world_size:
        raise ValueError(
            f"NEURON_RT_VISIBLE_CORES has {n} cores but world_size={world_size}"
        )
    per = n // world_size
    return [
        ",".join(str(c) for c in cores[r * per : (r + 1) * per])
        for r in range(world_size)
    ]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompts: list[str]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_new_tokens: int = 16
    return_logprob: bool = False
    generation_kwargs: dict[str, Any] = {}


class InitCommunicatorRequest(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5558


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _round_up_to_multiple(value: int, multiple: int, ceiling: int) -> int:
    """Round *value* up to the next ``multiple``, but no higher than *ceiling*."""
    if multiple <= 1:
        return min(value, ceiling)
    rounded = ((value + multiple - 1) // multiple) * multiple
    return min(rounded, ceiling)


@torch.inference_mode()
def _worker_generate(model, tokenizer, device, args, indexed_prompts, gen_args):
    """Run ``model.generate()`` on one shard; return list of per-completion dicts.

    Mirrors ``trl/trainer/grpo_trainer.py`` colocate path (L1430-1489):
    left-pad with ``pad_to_multiple_of``, ``GenerationConfig`` includes
    ``bos_token_id``, and completion masking uses the first-EOS index
    (not ``== pad_id``).
    """
    from transformers import GenerationConfig

    indices = [i for i, _ in indexed_prompts]
    prompts = [p for _, p in indexed_prompts]

    # Left padding so the last real prompt token sits at the same absolute
    # position for every row. ``pad_to_multiple_of`` keeps the prefill shape
    # stable across requests (required for torch.compile on Neuron). Capped
    # at ``--max-prompt-length`` via ``max_length`` + ``truncation=True``.
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        text=prompts,
        padding=True,
        padding_side="left",
        pad_to_multiple_of=args.pad_to_multiple,
        max_length=args.max_prompt_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    gen_kwargs = dict(gen_args.get("generation_kwargs") or {})
    cache_implementation = gen_kwargs.pop("cache_implementation", args.cache_implementation)

    gen_config = GenerationConfig(
        do_sample=True,
        num_return_sequences=gen_args["n"],
        temperature=gen_args["temperature"],
        top_p=gen_args["top_p"],
        top_k=gen_args["top_k"] if gen_args["top_k"] > 0 else 0,
        min_p=gen_args["min_p"],
        repetition_penalty=gen_args["repetition_penalty"],
        max_new_tokens=gen_args["max_new_tokens"],
        cache_implementation=cache_implementation,
        output_scores=gen_args["return_logprob"],
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )

    # Route torch.multinomial off the Neuron device for the sampling draw.
    # Softmax stays on device (preserving the compiled graph); only the RNG
    # step runs on CPU in fp32. Avoids Neuron's non-deterministic / bf16 sampler
    # biasing completions. Enabled by default; set HF_SERVE_CPU_MULTINOMIAL=0 to opt out.
    _use_cpu_mn = os.environ.get("HF_SERVE_CPU_MULTINOMIAL", "1") == "1"
    _orig_mn = torch.multinomial
    if _use_cpu_mn:
        def _cpu_mn(input, num_samples, replacement=False, generator=None, out=None):
            r = _orig_mn(
                input.detach().cpu().float(),
                num_samples,
                replacement=replacement,
                generator=generator,
            )
            return r.to(input.device)
        torch.multinomial = _cpu_mn
    try:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    finally:
        if _use_cpu_mn:
            torch.multinomial = _orig_mn

    prompt_len = input_ids.shape[1]
    sequences = out.sequences
    completions = sequences[:, prompt_len:]

    # Mask everything after the first EOS (colocate L1480-1485).
    eos_id = tokenizer.eos_token_id
    is_eos = completions == eos_id
    eos_idx = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
    )
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
        is_eos.size(0), -1
    )
    completion_mask = sequence_indices <= eos_idx.unsqueeze(1)

    # Strip via attention_mask / completion_mask (colocate L1486-1487), not
    # token-id equality — pad_token often equals eos_token so a comparison
    # against pad_id would drop legitimate EOS tokens from the prompt.
    prompt_ids_clean = [p[m].tolist() for p, m in zip(input_ids, attention_mask.bool())]
    completion_ids_trimmed = [
        c[m].tolist() for c, m in zip(completions, completion_mask)
    ]

    if gen_args["return_logprob"] and out.scores is not None:
        step_lps = [F.log_softmax(s.float(), dim=-1) for s in out.scores]
        token_logprobs_batched: list[list[list] | None] = []
        for row in range(completions.shape[0]):
            row_lps: list[list] = []
            for t, lp in enumerate(step_lps):
                token_id = int(completions[row, t].item())
                row_lps.append([float(lp[row, token_id].item()), token_id, None])
            token_logprobs_batched.append(row_lps)
    else:
        token_logprobs_batched = [None] * completions.shape[0]

    n = gen_args["n"]
    results: list[dict] = []
    for local_idx in range(input_ids.shape[0]):
        global_idx = indices[local_idx]
        for k in range(n):
            flat = local_idx * n + k
            comp = completion_ids_trimmed[flat]
            lp = token_logprobs_batched[flat]
            if lp is not None:
                lp = lp[: len(comp)]
            results.append(
                {
                    "index": global_idx,
                    "prompt_ids": prompt_ids_clean[local_idx],
                    "completion_ids": comp,
                    "token_logprobs": lp,
                }
            )

    if os.environ.get("HF_SERVE_PRINT_ROLLOUTS", "1") == "1":
        print(f"\n===== [ROLLOUT] n={len(results)} =====", flush=True)
        for r in results:
            p_text = tokenizer.decode(r["prompt_ids"], skip_special_tokens=True)
            c_text = tokenizer.decode(r["completion_ids"], skip_special_tokens=True)
            print(f"--- [{r['index']}] PROMPT ---\n{p_text}")
            print(f"--- [{r['index']}] COMPLETION ---\n{c_text}")
        print("===== [/ROLLOUT] =====\n", flush=True)

    return results


def _worker_main(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    core_env: str,
    req_q: "mp.Queue",
    res_q: "mp.Queue",
    wgt_q: "mp.Queue",
    wgt_ack_q: "mp.Queue",
    ready_q: "mp.Queue",
):
    if core_env:
        os.environ["NEURON_RT_VISIBLE_CORES"] = core_env

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for causal LM generation.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_implementation
    )
    device = torch.device(args.device)
    model = model.to(device).eval()

    if device.type == "privateuseone" or os.environ.get("ON_NEURON") == "1":
        logger.info(f"[worker {rank}] compiling model.forward (backend='neuron')")
        model._rollout_forward = torch.compile(
            model.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        model.forward = model._rollout_forward
    else:
        logger.info(f"[worker {rank}] skipping torch.compile (non-Neuron device)")

    param_map = dict(model.named_parameters())
    ready_q.put(rank)
    logger.info(
        f"[worker {rank}] ready on {device}, cores={core_env or '(unset)'}, "
        f"{len(param_map)} parameters"
    )

    while True:
        # Drain any pending weight updates first so sync can't be starved by generate traffic.
        while True:
            try:
                msg = wgt_q.get_nowait()
            except Empty:
                break
            kind = msg[0]
            if kind == "update":
                _, name, dtype_str, shape, shm_name, nbytes = msg
                param = param_map.get(name)
                if param is None:
                    wgt_ack_q.put((rank, "error", f"unknown param {name}"))
                    continue
                try:
                    t0 = time.time()
                    shm = shared_memory.SharedMemory(name=shm_name)
                    try:
                        dtype = _parse_torch_dtype(dtype_str)
                        tensor = torch.frombuffer(
                            shm.buf[:nbytes], dtype=dtype
                        ).reshape(tuple(shape))
                        with torch.no_grad():
                            param.data.copy_(tensor.to(param.device, dtype=param.dtype))
                        del tensor
                    finally:
                        shm.close()
                    dt = time.time() - t0
                    if nbytes > 10_000_000:
                        logger.info(
                            f"[worker {rank}] applied {name} "
                            f"({nbytes / 1e6:.1f} MB) in {dt:.2f}s"
                        )
                    wgt_ack_q.put((rank, "ok", None))
                except Exception as exc:
                    logger.exception(f"[worker {rank}] weight update failed for {name}")
                    wgt_ack_q.put((rank, "error", str(exc)))
            elif kind == "shutdown":
                return

        try:
            req = req_q.get(timeout=0.1)
        except Empty:
            continue

        kind = req[0]
        if kind == "generate":
            _, req_id, indexed_prompts, gen_args = req
            try:
                result = _worker_generate(
                    model, tokenizer, device, args, indexed_prompts, gen_args
                )
                res_q.put((req_id, rank, "ok", result))
            except Exception as exc:
                logger.exception(f"[worker {rank}] generate failed")
                res_q.put((req_id, rank, "error", str(exc)))
        elif kind == "shutdown":
            return


# ---------------------------------------------------------------------------
# Router state (rank 0 / main process)
# ---------------------------------------------------------------------------


class RouterState:
    world_size: int = 1
    req_qs: list = []
    res_q: Any = None
    wgt_qs: list = []
    wgt_ack_q: Any = None
    procs: list = []

    weight_sync_lock = threading.Lock()
    zmq_thread: threading.Thread | None = None
    zmq_stop = threading.Event()
    req_counter = itertools.count()
    wgt_sync_counter: int = 0
    args: argparse.Namespace | None = None


state = RouterState()

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    with state.weight_sync_lock:
        return _route_generate(req)


def _route_generate(req: GenerateRequest) -> dict:
    N = state.world_size
    # Round-robin shard; preserves global prompt index for reassembly on return.
    shards: list[list[tuple[int, str]]] = [[] for _ in range(N)]
    for i, p in enumerate(req.prompts):
        shards[i % N].append((i, p))

    gen_args = {
        "n": req.n,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "min_p": req.min_p,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
        "return_logprob": req.return_logprob,
        "generation_kwargs": dict(req.generation_kwargs or {}),
    }

    req_id = next(state.req_counter)
    nonempty = 0
    for r, shard in enumerate(shards):
        if not shard:
            continue
        state.req_qs[r].put(("generate", req_id, shard, gen_args))
        nonempty += 1

    all_results: list[dict] = []
    errors: list[str] = []
    got = 0
    while got < nonempty:
        rid, rank, status, payload = state.res_q.get()
        if rid != req_id:
            continue  # stale; the serialize-on-lock invariant should prevent this.
        if status == "ok":
            all_results.extend(payload)
        else:
            msg = f"rank {rank}: {payload}"
            logger.error(f"[generate] worker failed: {msg}")
            errors.append(msg)
        got += 1

    if errors:
        # Surface the failure via HTTP status; a silent empty ``results`` list
        # was confusing callers.
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500, detail={"message": "; ".join(errors)}
        )

    all_results.sort(key=lambda r: r["index"])
    return {"results": all_results}


# ---------------------------------------------------------------------------
# Weight sync (ZMQ REP) — rank 0 fans each update out to all workers
# ---------------------------------------------------------------------------


def _zmq_rep_loop(port: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ REP bound to tcp://*:{port}")

    try:
        while not state.zmq_stop.is_set():
            if sock.poll(timeout=500) == 0:
                continue
            try:
                meta_bytes, raw = sock.recv_multipart()
            except zmq.error.Again:
                continue
            meta = msgpack.unpackb(meta_bytes)

            if meta.get("cmd") == "done":
                sock.send(msgpack.packb({"status": "ok"}))
                break

            try:
                name = meta["name"]
                dtype_str = meta["dtype"]
                shape = list(meta["shape"])
                nbytes = len(raw)

                # Stage the raw bytes into a POSIX shared-memory segment once;
                # every worker reads the same segment without extra copies.
                # Avoids the N× pickle through mp.Queue that made large params
                # look like a hang.
                t0 = time.time()
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
                shm.buf[:nbytes] = raw

                state.wgt_sync_counter += 1
                if state.wgt_sync_counter == 1 or state.wgt_sync_counter % 50 == 0:
                    logger.info(
                        f"[weight sync] #{state.wgt_sync_counter} {name} "
                        f"({nbytes / 1e6:.1f} MB) → {state.world_size} worker(s)"
                    )

                try:
                    for q in state.wgt_qs:
                        q.put(("update", name, dtype_str, shape, shm.name, nbytes))

                    errors: list[str] = []
                    for ack_i in range(state.world_size):
                        try:
                            r, status, msg = state.wgt_ack_q.get(timeout=120.0)
                        except Empty:
                            errors.append(
                                "timeout waiting for worker ack "
                                f"(param={name}); a worker may be hung or dead"
                            )
                            break
                        if status != "ok":
                            errors.append(f"rank {r}: {msg}")
                finally:
                    # All workers have ack'd (or timed out) — safe to unlink.
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass

                elapsed = time.time() - t0
                if errors:
                    sock.send(
                        msgpack.packb({"status": "error", "message": "; ".join(errors)})
                    )
                else:
                    sock.send(msgpack.packb({"status": "ok"}))
                    if nbytes > 10_000_000:  # log big params so progress is visible
                        logger.info(
                            f"[weight sync] {name} done in {elapsed:.2f}s "
                            f"({nbytes / 1e6 / elapsed:.1f} MB/s/worker)"
                        )
            except Exception as exc:
                logger.exception(f"Weight update failed for {meta.get('name')}")
                sock.send(msgpack.packb({"status": "error", "message": str(exc)}))
    finally:
        sock.close(linger=0)
        logger.info("ZMQ REP socket closed")


@app.post("/init_communicator/")
def init_communicator(req: InitCommunicatorRequest):
    if state.zmq_thread is not None and state.zmq_thread.is_alive():
        return {"status": "error", "message": "weight sync session already active"}

    state.weight_sync_lock.acquire()
    state.zmq_stop.clear()
    state.wgt_sync_counter = 0
    state.zmq_thread = threading.Thread(
        target=_zmq_rep_loop, args=(req.port,), daemon=True
    )
    state.zmq_thread.start()
    return {"status": "ok"}


@app.post("/close_communicator/")
def close_communicator():
    state.zmq_stop.set()
    if state.zmq_thread is not None:
        state.zmq_thread.join(timeout=30)
        state.zmq_thread = None
    if state.weight_sync_lock.locked():
        state.weight_sync_lock.release()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--zmq-port", type=int, default=5558)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--device", default="neuron")
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument(
        "--pad-to-multiple",
        type=int,
        default=128,
        help="Round each request's prompt-pad length up to this multiple "
        "(capped at --max-prompt-length). Use 1 to disable bucketing.",
    )
    p.add_argument("--cache-implementation", default="static")
    p.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of data-parallel worker processes. Each worker gets an equal slice "
        "of NEURON_RT_VISIBLE_CORES.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    state.args = args

    visible_env = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if visible_env:
        cores = _parse_core_range(visible_env)
        core_envs = _slice_cores(cores, args.world_size)
        logger.info(
            f"NEURON_RT_VISIBLE_CORES={visible_env} -> per-rank cores={core_envs}"
        )
    else:
        core_envs = ["" for _ in range(args.world_size)]
        logger.info(
            "NEURON_RT_VISIBLE_CORES not set; workers inherit parent visibility"
        )

    ctx = mp.get_context("spawn")
    state.world_size = args.world_size
    state.req_qs = [ctx.Queue() for _ in range(args.world_size)]
    state.res_q = ctx.Queue()
    state.wgt_qs = [ctx.Queue() for _ in range(args.world_size)]
    state.wgt_ack_q = ctx.Queue()
    ready_q = ctx.Queue()

    procs = []
    for r in range(args.world_size):
        proc = ctx.Process(
            target=_worker_main,
            args=(
                r,
                args.world_size,
                args,
                core_envs[r],
                state.req_qs[r],
                state.res_q,
                state.wgt_qs[r],
                state.wgt_ack_q,
                ready_q,
            ),
            daemon=False,
        )
        proc.start()
        procs.append(proc)
    state.procs = procs

    logger.info(f"Waiting for {args.world_size} worker(s) to signal ready...")
    ready: set[int] = set()
    while len(ready) < args.world_size:
        try:
            r = ready_q.get(timeout=1.0)
            ready.add(r)
            logger.info(f"  worker {r} ready ({len(ready)}/{args.world_size})")
        except Empty:
            # Surface a dead worker immediately instead of hanging forever.
            dead = [i for i, p in enumerate(procs) if not p.is_alive()]
            if dead:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(
                    f"Worker rank(s) {dead} died before signaling ready. "
                    "Check the server log for the traceback."
                )

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        for q in state.req_qs:
            try:
                q.put(("shutdown",))
            except Exception:
                pass
        for p in procs:
            p.join(timeout=10)


if __name__ == "__main__":
    main()
