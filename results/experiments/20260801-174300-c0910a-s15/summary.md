# Context-length probe matrix

NVIDIA B200, batch size 1, 256-token completion, probe layer 24, BF16. Times are
the median of three measured runs after an untimed warmup. Peak memory includes
the approximately 57 GiB resident model weights.

| input | output | vLLM Original | Sonic checkpointing | Sonic no checkpointing |
|---:|---:|---:|---:|---:|
| 2,048 | 256 | 0.783 s / 159.90 GiB | 0.154 s / 58.16 GiB | 0.098 s / 61.67 GiB |
| 4,096 | 256 | 0.787 s / 160.07 GiB | 0.199 s / 58.58 GiB | 0.164 s / 65.58 GiB |
| 8,192 | 256 | 0.870 s / 160.32 GiB | 0.398 s / 59.47 GiB | 0.301 s / 73.46 GiB |
| 16,384 | 256 | 1.051 s / 160.35 GiB | 1.097 s / 61.23 GiB | 0.842 s / 89.18 GiB |

Probe time as a fraction of the vLLM 256-token completion time:

| input | Sonic checkpointing | Sonic no checkpointing |
|---:|---:|---:|
| 2,048 | 19.7% | 12.5% |
| 4,096 | 25.3% | 20.9% |
| 8,192 | 45.8% | 34.6% |
| 16,384 | 104.4% | 80.1% |

The roughly 160 GiB vLLM peak is not per-request demand. vLLM was configured
with `gpu_memory_utilization=0.9` and preallocated approximately 102 GiB for its
KV-cache pool in addition to the weights. It therefore must not be interpreted
as evidence that Original intrinsically needs more memory than the probe.

The first Sonic attempts at 2,048 and 8,192 tokens failed before measurement
because QuACK tried to create new shape-specific cache entries under a full Home
quota. They were rerun successfully with the QuACK cache on node-local `/tmp`;
those successful records replaced the error records in `probe_matrix.json`.
