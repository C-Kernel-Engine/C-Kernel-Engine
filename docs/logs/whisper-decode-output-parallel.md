# Whisper decode FP32 output parallelism

## Scope

The FP32 llama-production GEMM provider previously forced every single-token
projection through its serial leaf. This change allows sufficiently large
single-token projections to partition independent output dots over CKE's
persistent thread pool. Each output remains owned by one worker, so its K
reduction and arithmetic order are unchanged.

Work below 512 KiFMA remains serial. On the Intel Core i7-14700T P3 with 20
threads, the `M=1, N=512, K=512` projection regressed when dispatched, while
the three larger Whisper shapes improved. The threshold retains the former
path for that small shape.

## Kernel evidence

The production dispatch, serial leaf, manually partitioned output ranges, and
the llama.cpp GGML CPU graph were bit exact for all tested values. The matrix
covered Whisper's four decode shapes plus the existing narrow decode, prefill,
and Qwen3.5 router cases at 1, 16, 20, and 24 threads.

P3 microbenchmarks at 20 threads:

| Shape (`M=1`) | Serial | Parallel dispatch | Speedup |
|---|---:|---:|---:|
| `N=512, K=512` | 0.020 ms | 0.023 ms | 0.85x; retained serial |
| `N=2048, K=512` | 0.129 ms | 0.026 ms | 5.00x |
| `N=512, K=2048` | 0.077 ms | 0.030 ms | 2.60x |
| `N=51865, K=512` | 5.150 ms | 1.721 ms | 2.99x |

## Five-minute Whisper-base evidence

The retained five-minute 16 kHz fixture was run three times with the
FP32-preserved `openai/whisper-base` decoder and 20 threads. All runs matched
the request-scoped-frontend baseline exactly: 974 token IDs, 11 window
boundaries, and every feature and encoder hash.

| Measurement | Frontend-reuse baseline | Output-parallel median |
|---|---:|---:|
| Decoder generation | 12.673 s | 9.871 s |
| Decoder prefill | 1.096 s | 1.148 s |
| Encoder | 10.415 s | 10.563 s |
| Wall time | 29.04 s | 26.45 s |

Decoder generation fell 22.1%. Wall time is not directly additive with the
separate parallel-GELU branch; combined evidence requires a merged-candidate
replay. Attention and per-window process/runtime initialization remain
separate utilization targets.

## Full-track evidence

The complete 42:22 recording also matched the frontend-reuse baseline exactly:
9,870 token IDs, 92 source windows, and every feature and encoder hash. Decoder
generation fell from 128.635 to 99.556 seconds (22.6%), while wall time fell
from 263.66 to 234.80 seconds (10.9%). Decoder prefill changed from 9.616 to
9.119 seconds. The candidate used no swap and peaked at 543 MiB RSS.

Average process occupancy was 615%, or about 6.15 core equivalents. Output
parallelism closes one measured serial region but does not establish complete
decoder utilization. Attention and per-window worker/runtime initialization
remain visible follow-up work.
