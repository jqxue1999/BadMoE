# PIRA Timing and Memory Benchmarks

This directory contains the minimal code used for the rebuttal timing
measurements. It does not run attacks, defense evaluation, or safety scoring.
A seeded synthetic output direction is used so that the PIRA probe has the same
autograd graph and tensor shapes as the real method without requiring extracted
safety artifacts.

## Files

- `scripts/moe/rebuttal_cost_benchmark.py`: batch-1 Original/PIRA timing,
  including probe time, prefill time, TTFT, total generation time, and CUDA
  peak memory.
- `scripts/moe/rebuttal_batch_benchmark.py`: fixed-resource batched
  Original/PIRA comparison with independent probe microbatching.
- `scripts/moe/rebuttal_vllm_baseline.py`: Original-only vLLM reference.
- Matching `.slurm` launchers are provided for each benchmark.

`rebuttal_batch_benchmark.py` imports the PIRA hooks and probe implementation
from `rebuttal_cost_benchmark.py`, so keep both files together.

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
