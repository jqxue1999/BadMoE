# PIRA Timing and Memory Benchmarks

This directory contains the minimal code used for the rebuttal timing
measurements. It does not run attacks, defense evaluation, or safety scoring.
A seeded synthetic output direction is used so that the PIRA probe has the same
autograd graph and tensor shapes as the real method without requiring extracted
safety artifacts.

## Files

Probe implementation and its correctness tests:

- `scripts/moe/pira_probe.py`: the probe. `ReferenceProbe` is the unoptimized
  implementation; `FastProbe` applies layer truncation, gradient checkpointing,
  and a memory-efficient attention backward. Every optimization is
  mathematically neutral -- there is no closed form, no surrogate direction, and
  no dropped term, only a better execution of the same autograd graph.
- `scripts/moe/test_probe_equivalence_cpu.py`: **run this first.** Tiny CPU MoE,
  no GPU needed, seconds to run. In fp32 it requires the optimized probe to be
  *bitwise* identical to the reference.
- `scripts/moe/test_probe_equivalence.py`: the same check on the real model,
  ablating one optimization at a time. bf16 admits small numeric differences, so
  the binding criteria are an identical top-K suppression set and Spearman 1.0.
- `scripts/moe/probe_workload.py`: shared model loading and prompt construction,
  so every benchmark builds identical batches.

Measurement:

- `scripts/moe/rebuttal_probe_scaling.py`: probe time and peak activation memory
  across prompt lengths and batch sizes, with and without checkpointing.
- `scripts/moe/rebuttal_vllm_pira_benchmark.py`: end-to-end vLLM comparison of
  unbiased vs PIRA-biased generation, over an (input, output, concurrency) grid.
  This measures the routing-intervention cost instead of assuming it is zero.
- `scripts/moe/rebuttal_true_overhead.py`: combines the engine baseline, the
  probe, and the measured routing cost into `True Total`.
- `scripts/moe/pira_vllm_routing.py`: applies PIRA's per-request, pre-softmax
  router bias inside vLLM's expert-selection path, keyed by engine request id so
  it stays correct under continuous batching. Includes CPU self-checks
  (`python scripts/moe/pira_vllm_routing.py`) that need neither GPU nor vLLM.
- `scripts/moe/rebuttal_cost_benchmark.py`: batch-1 Original/PIRA timing,
  including probe time, prefill time, TTFT, total generation time, and CUDA
  peak memory.
- `scripts/moe/rebuttal_batch_benchmark.py`: fixed-resource batched
  Original/PIRA comparison with independent probe microbatching.
- `scripts/moe/rebuttal_vllm_baseline.py`: Original-only vLLM reference.
- Matching `.slurm` launchers are provided for each benchmark.

`rebuttal_batch_benchmark.py` imports the PIRA hooks and probe implementation
from `rebuttal_cost_benchmark.py`, so keep both files together.

## Cost Model

PIRA's cost has two parts, and **both are measured**:

    PIRA     = [probe: forward+backward over layers 0..L] + [prefill + decode under biased routing]
    Original =                                              [prefill + decode]

    True Total = Original turnaround on a serving engine at maximum GPU utilization
               + standalone PIRA forward+backward probe
               + measured cost of generating under biased routing

The third term is not assumed to be zero. Biased routing leaves the FLOP count
identical -- top-k is unchanged and the expert MLPs are untouched, only *which*
experts a token reaches -- but identical FLOPs do not imply identical latency:
suppressing a request's experts changes the expert-token distribution, and with it
grouped-GEMM shapes, load balance across experts, and kernel scheduling. Whether
that costs anything measurable is an empirical question, so
`rebuttal_vllm_pira_benchmark.py` measures unbiased vs biased generation on the
same engine and `rebuttal_true_overhead.py` folds the result in. Run the combiner
without `--routing-json` and it labels its output a lower bound rather than
quietly reporting zero.

The overhead *ratio* depends on how prompt-heavy the workload is, so results are
reported over a grid of input length, output length, and concurrency rather than
as one number.

Adding a bias to router logits before top-k is already a first-class operation in
vLLM (`e_score_correction_bias`, used by DeepSeek-V3), but it is per-expert,
post-softmax, and selection-only. PIRA needs a per-request, pre-softmax variant
whose gate weights come from the biased distribution, which is what
`pira_vllm_routing.py` supplies.

## Running PIRA Inside vLLM

vLLM V1 runs the model in an EngineCore worker process, so the routing hooks are
installed there via `collective_rpc`, not around `llm.generate()` in the driver:

```python
import pira_vllm_routing as pira

# Layers are explicit and required. They cannot be inferred from registered
# biases, because installation necessarily precedes any request existing.
layers = pira.install_in_worker(llm, range(probe_layer + 1), beta=10.0, strict=True)
print(pira.verify_hooks(llm))          # hooks sit on the live router objects?

for cell in cells:
    pira.reset_counters(llm)           # per-cell, so no cell hides behind another
    request_ids = llm.enqueue(prompts, sampling)   # real engine ids, not yet running
    pira.register_biases_in_worker(llm, biases_by_request_id)
    outputs = llm.wait_for_completion()            # now execute
    assert pira.worker_diagnostics(llm)["rows_biased"] > 0

pira.uninstall_in_worker(llm)
```

Four details that are easy to get wrong:

- **The layer set must be passed in.** Deriving it from the registered biases
  installs nothing: the engine assigns request ids only when requests are
  enqueued, which is necessarily after the hooks exist.
- **Request ids come from `enqueue()`.** It returns the ids the engine actually
  assigned without starting execution, so biases can be registered against them
  before the first forward pass. Guessing the id scheme would silently leave every
  row unbiased and make PIRA look free.
- **Biases are keyed by request id, never by batch position.** Under continuous
  batching the scheduler swaps and condenses rows (`InputBatch.swap_states`,
  `condense`), so a position-indexed table would start applying one request's
  suppression set to another. The engine's `req_ids` order is re-read every
  forward pass. `verify_reordering()` tests exactly this and fails a
  position-indexed implementation.
- **The hook delegates to vLLM's own selector.** It adds the bias to
  `router_logits` and calls the original `select_experts`, so the engine keeps its
  fused routing kernel, EPLB mapping, and indices-dtype handling. Reimplementing
  softmax and top-k in Python would make the benchmark measure the prototype
  rather than the method. Bias matrices are built on-device and cached per
  (layer, row order); diagnostics are counted from host-side values so they add no
  device synchronization to the timed path.

`worker_diagnostics()` reports `rows_biased` **per cell**. A cell with
`rows_biased == 0` measured the Original model twice; the benchmark exits nonzero
and names those cells, and the combiner drops them.

## Binding Rebuttal Workload

| Axis | Values |
|---|---|
| Input length | 128, 1024, 4096 (plus 8192 in `MODE=longctx`) |
| Output length | 128, 256 |
| Concurrency | 1, 8, 32 |
| Requests per cell | 32 |
| Repeats | 3, median reported |
| Model / GPU | Qwen3-30B-A3B, BF16, one B200 |
| Probe layer / K / beta | 24, 25, 10.0 |

128/128 at concurrency 1 is the latency-sensitive corner; 4096/128 is the
prompt-heavy corner where the probe is most expensive relative to generation;
concurrency 32 is where routing-induced load imbalance would show up.

**Concurrency is enforced, not just labelled.** Each cell runs
`requests_per_cell / concurrency` sequential waves of exactly `concurrency`
requests and times the whole set. Submitting all 32 requests at once would let the
scheduler keep up to `max_num_seqs` sequences active regardless of the label, so
the "concurrency 1" and "concurrency 8" rows would both have measured 32.

Because each routing cell measures Original and biased generation back to back on
one engine at one concurrency, `rebuttal_true_overhead.py` prefers those
same-cell numbers over the standalone baseline:

    true_total = measured biased generation + probe
               = Original + probe + (biased - Original)

all from the same cell. The standalone `rebuttal_vllm_baseline.py` run remains
useful as an independent check that the harness reaches the engine's normal
throughput, but it is not needed for the equation.

## Probe Optimizations

Each is exactness-preserving, and `test_probe_equivalence*.py` verifies that
rather than assuming it:

| Optimization | Effect | Why it cannot change the result |
|---|---|---|
| Layer truncation | Skips layers above `L` | Layers above `L` are not in the graph of `s(q)`, so they cannot influence any gradient |
| Gradient checkpointing | Peak activation memory becomes `O(1)` in depth instead of `O(L)` | Recomputes the same deterministic activations it would otherwise have stored |
| Memory-efficient attention backward | Attention workspace `O(S)` instead of `O(S^2)` | A different kernel for the same mathematical operation |
| Batching, early graph release | Better utilization; bounded memory | Scheduling only |

Deliberately **not** used: closed-form gradients, forward-only surrogates, or
any approximation of the backward pass. A forward-only surrogate that replaces
the adjoint with the readout direction was tested and rejected -- it reorders
experts (Spearman about 0.3 on a controlled setup) and would select a different
suppression set.

## Measured Execution Path

For each PIRA request batch, the code performs:

1. A differentiable prompt forward pass truncated after the selected probe
   layer.
2. `torch.autograd.grad` with respect to per-request router-bias leaves. Model
   parameters remain frozen.
3. Per-request top-K expert selection and release of the probe graph.
4. A fresh biased prefill, where the bias is applied before router softmax and
   top-k, followed by exact-length KV-cached greedy decoding.

The Original and PIRA measurements use the same Hugging Face model path,
precision, prompt, generation batch size, and GPU. Probe microbatch size can be
smaller than generation batch size: each probe graph is released before the
next microbatch, while the resulting small router-bias tensors are concatenated
for one larger generation batch.

## Environment

The reported runs used:

- One NVIDIA B200
- Qwen/Qwen3-30B-A3B in BF16
- Python 3.12
- PyTorch 2.10.0
- Transformers 5.6.2 for the Hugging Face runs
- vLLM 0.19.0 for the Original-only engine reference

Create separate environments because vLLM pins its own compatible
Transformers stack:

```bash
uv venv --python 3.12 .venv-hf
uv pip install --python .venv-hf/bin/python \
  torch==2.10.0 transformers==5.6.2 accelerate==1.13.0 safetensors==0.7.0

uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.19.0
```

## Recommended Run Order

Correctness before timing: a speedup only means something once the fast probe is
shown to compute the same thing as the slow one.

```bash
# 0. No GPU needed. Seconds. Run before submitting anything.
python scripts/moe/test_probe_equivalence_cpu.py   # probe gradients bitwise-exact
python scripts/moe/pira_vllm_routing.py            # biased selection + reordering safety

# 1. Equivalence on the real model. Must pass before quoting any timing.
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=equivalence,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_probe_scaling.slurm

# 1b. Strictest variant: fp32, deterministic kernels.
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=equivalence-strict,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_probe_scaling.slurm

# 2. Long-context scaling: probe time and activation memory, 128..8192 tokens,
#    with and without checkpointing.
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=scaling,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_probe_scaling.slurm

# 3. Speedup of the optimized probe over the unoptimized reference.
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=scaling-with-reference,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_probe_scaling.slurm

# 4. Engine baseline at maximum GPU utilization (vLLM env).
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=long,PYTHON_BIN=$PWD/.venv-vllm/bin/python \
  scripts/moe/rebuttal_vllm_baseline.slurm

# 5. Routing cost: Original vs biased generation on the same engine (vLLM env).
#    Run MODE=smoke first and confirm rows_biased > 0 before the full grid.
sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=smoke,PYTHON_BIN=$PWD/.venv-vllm/bin/python \
  scripts/moe/rebuttal_vllm_pira_benchmark.slurm

sbatch --partition=<p> --account=<a> --qos=<q> \
  --export=ALL,MODE=grid,PYTHON_BIN=$PWD/.venv-vllm/bin/python \
  scripts/moe/rebuttal_vllm_pira_benchmark.slurm

# 6. Combine all three into True Total.
python scripts/moe/rebuttal_true_overhead.py \
  --engine-json  timing_results/vllm_long_<jobid>.json \
  --probe-json   timing_results/probe_scaling_<jobid>.json \
  --routing-json timing_results/vllm_pira_grid_<jobid>.json \
  --output       timing_results/true_overhead.json
```

Step 2 sweeps prompt lengths, so run it with enough walltime; the 8192-token
cell without checkpointing is expected to be the one that OOMs, which is itself
the result being reported.

## Slurm Usage

The launchers intentionally omit cluster-specific account, QOS, and partition
values. Supply them to `sbatch`. GPU inference must run on a compute node, not
on a login node.

Long-completion HF comparison ($N=16$, input 128, output 256, batches 1/4/16):

```bash
sbatch --partition=<gpu-partition> --account=<account> --qos=<qos> \
  --export=ALL,MODE=long,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_batch_benchmark.slurm
```

High-concurrency HF comparison ($N=512$, input/output 128, generation batch
512) with a probe microbatch of 64:

```bash
sbatch --partition=<gpu-partition> --account=<account> --qos=<qos> \
  --export=ALL,MODE=microstress128,PROBE_MICROBATCH_SIZE=64,PYTHON_BIN=$PWD/.venv-hf/bin/python \
  scripts/moe/rebuttal_batch_benchmark.slurm
```

Original-only vLLM references for the same two workloads:

```bash
sbatch --partition=<gpu-partition> --account=<account> --qos=<qos> \
  --export=ALL,MODE=long,PYTHON_BIN=$PWD/.venv-vllm/bin/python \
  scripts/moe/rebuttal_vllm_baseline.slurm

sbatch --partition=<gpu-partition> --account=<account> --qos=<qos> \
  --export=ALL,MODE=stress,PYTHON_BIN=$PWD/.venv-vllm/bin/python \
  scripts/moe/rebuttal_vllm_baseline.slurm
```

Set `CUDA_MODULE`, `HF_HOME`, `TORCH_HOME`, or `OUTPUT_DIR` through
`--export` if required by the cluster. JSON results default to
`timing_results/`, which is ignored by Git.

## Rebuttal Measurements

For $N=16$, 128-token prompts, and 256-token completions:

| Batch | vLLM Original (s) | HF Original (s) | HF PIRA (s) | Paired HF overhead |
|---:|---:|---:|---:|---:|
| 1 | 16.13 | 146.63 | 151.21 | 3.1% |
| 4 | 5.59 | 36.93 | 38.15 | 3.3% |
| 16 | 1.94 | 9.21 | 9.48 | 2.9% |

For $N=512$, 128-token prompts and completions, and generation batch 512:

| Method | Probe microbatch | Total (s) | Requests/s | Peak allocated memory |
|---|---:|---:|---:|---:|
| vLLM Original | - | 2.53 | 202.1 | 90% GPU budget configured |
| HF Original | - | 5.75-5.81 | 88.1-89.1 | 71.94-71.95 GiB |
| HF PIRA | 256 | 7.63 | 67.1 | 152.28 GiB |
| HF PIRA | 128 | 7.55 | 67.9 | 104.59 GiB |
| HF PIRA | 64 | 7.74 | 66.1 | 80.76 GiB |

The vLLM row is an Original-only serving-engine reference. vLLM preallocates
its KV cache, so its configured memory budget is not directly comparable to
PyTorch `max_memory_allocated`. Current vLLM execution uses an inference-only
model path and does not expose the differentiable per-request router state
needed by PIRA. Consequently, PIRA overhead is computed only from paired HF
Original/PIRA runs; dividing vLLM Original time by HF PIRA time would also
include fused-kernel, paged-attention, CUDA-graph, and scheduler differences.
