# Grouped-GEMM probe analysis

Run: `20260801-034408-c0903a-s15`

## Summary

The isolated one-layer grouped-GEMM microbenchmark is substantially faster than its synthetic HF-loop reference, but that speedup does **not** carry over to the full real-model PIRA probe. In the full probe, grouped-GEMM is 15.8%--27.8% slower than the HF path and is not numerically equivalent. Therefore the isolated result should not currently be used as evidence that grouped-GEMM accelerates end-to-end PIRA.

## Observed results

### Isolated MoE microbenchmark

| Sequence length | HF forward+backward | Grouped-GEMM | Reported speedup |
| ---: | ---: | ---: | ---: |
| 128 | 35.714 ms | 1.510 ms | 23.6x |
| 1024 | 36.655 ms | 1.722 ms | 21.3x |
| 4096 | 37.289 ms | 3.939 ms | 9.5x |

### Full real-model probe

| Prompt length | HF forward+backward | Grouped-GEMM | Grouped/HF speedup | Relative difference | Top-k overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 112.546 ms | 129.729 ms | 0.868x | 0.221 | 0.76 |
| 1024 | 108.520 ms | 138.637 ms | 0.783x | 0.212 | 0.92 |
| 4096 | 230.433 ms | 275.219 ms | 0.837x | 0.234 | 0.92 |

The grouped path also materializes approximately 28.125 GiB of stacked expert weights. Peak memory was about 85--86.5 GiB, compared with an initial HF peak of about 57 GiB (later HF readings retain allocator state from the grouped-weight caches).

The detailed machine-readable measurements are in `probe_backends.json` and `probe_scaling.json` in this directory.

## Code-confirmed causes of the microbenchmark/full-probe gap

1. **The isolated HF reference is weaker than the model's fused HF implementation.** The isolated `moe_hf_loop` computes gate and up projections with two separate matrix multiplications and uses out-of-place `out = out.index_add(...)`. The actual Qwen HF path uses one combined `gate_up_proj` linear operation, chunks its result, and uses in-place `index_add_`.

2. **The grouped implementation loses the fused gate/up projection.** For the model's fused weight layout, it splits the combined gate/up weights and performs separate grouped GEMMs. The HF path therefore needs a combined gate/up projection plus a down projection, whereas the current grouped path performs gate, up, and down separately.

3. **Routing preparation adds synchronization and sorting overhead.** The grouped path performs a stable `argsort`, `bincount`, and `counts.to(torch.int64).cpu()`. The `.cpu()` transfer introduces a GPU-to-CPU synchronization. This work occurs per MoE layer and is replayed during checkpointed backward recomputation.

4. **The microbenchmark covers only one synthetic MoE layer.** The full probe includes attention, normalization, routers, residual operations, activation checkpoint replay, and 25-layer accumulation. Even a real MoE-kernel improvement would be bounded by the fraction of total time spent in that kernel.

5. **The paths are not numerically equivalent.** Relative differences of 0.21--0.23 and non-identical top-k selections mean the timings are not yet an apples-to-apples comparison. Small per-layer layout or precision differences can compound through 25 layers and the backward pass.

## Hypotheses to validate

These are proposed explanations/optimizations, not yet established conclusions:

- Preserve the fused gate/up representation and issue one grouped GEMM with output width `2 * intermediate_size`, followed by a chunk, instead of two grouped GEMMs.
- Keep expert counts on-device, or cache/reuse routing metadata when the computation permits it, to remove the per-layer `.cpu()` synchronization.
- Profile stable sorting and permutation construction separately. A different grouped-GEMM API or routing representation may accept device-side offsets and avoid repeated sorting.
- Compare one real model layer first, then progressively increase the number of layers. This should reveal where numerical drift first appears and how routing changes amplify it.
- Compare identical saved routing assignments and identical dtypes for HF and grouped kernels before measuring speed. Kernel performance claims should wait until output and gradient tolerances pass.
- Separate one-time stacked-weight construction and allocator retention from steady-state latency and peak-memory reporting.

## Suggested next experiments

1. Add output and gradient equivalence assertions for a single real MoE block, including both module-list and fused layouts.
2. Benchmark the real fused HF block against (a) the current three-GEMM grouped path and (b) a fused gate/up two-GEMM grouped path.
3. Use a GPU profiler to quantify grouped GEMM time, sort/permutation time, host synchronization, and checkpoint replay separately.
4. Repeat the full probe only after the single-block equivalence test passes, and report both latency and the extra stacked-weight memory.

## Current conclusion

The 9.5x--23.6x isolated result primarily shows that grouped-GEMM beats the current synthetic per-expert Python-loop reference. It does not show a speedup over the optimized fused HF model path. On this B200 run, the full grouped-GEMM probe is slower, uses substantially more memory, and changes model behavior; it should remain an experimental backend until equivalence and routing overhead are fixed.
