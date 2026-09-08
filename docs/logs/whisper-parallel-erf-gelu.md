# Whisper exact GELU parallelization

Date: 2026-09-08

## Scope

Whisper's FP32 encoder uses the exact ERF GELU formula. The previous provider
evaluated every element serially even though elements are independent. This
change adds an explicitly selected thread-pool provider. Each worker owns a
disjoint contiguous output range and preserves the scalar FP64 `erf` formula
and its single FP32 output store.

Other GELU contracts and model circuits retain their existing providers.

## Numerical evidence

The native oracle compares the parallel provider with the scalar formula and
requires byte equality. The real Whisper-base runs also matched the merged
baseline exactly for:

- all 974 token IDs and 11 windows in three five-minute runs;
- all 9,870 token IDs and 92 windows in the 42:22 run;
- every per-window feature hash and encoder-output hash;
- every source-frame window boundary.

## Isolated P3 measurements

Median of five measured calls after warmup, using 20 threads:

| Elements | Scalar | Parallel | Speedup |
| ---: | ---: | ---: | ---: |
| 768,000 | 20.27 ms | 1.67 ms | 12.17x |
| 1,536,000 | 40.45 ms | 3.11 ms | 12.99x |
| 3,072,000 | 81.42 ms | 5.99 ms | 13.60x |

All compared outputs were byte-identical. One-thread dispatch retained scalar
performance, while 4, 8, and 16 threads also produced byte-identical outputs.

## End-to-end P3 measurements

Whisper-base, 20 CK threads, timestamps enabled, maximum 448 tokens per
window:

| Fixture | Baseline wall | Parallel wall | Baseline encoder | Parallel encoder |
| --- | ---: | ---: | ---: | ---: |
| 5:00 video | 29.04 s median | 24.33 s median | 10.45 s | 5.56 s median |
| 42:22 video | 263.66 s | 223.93 s | 87.56 s | 46.55 s |

The complete recording consumed all 2,542 seconds in 92 contiguous windows.
Its run reported 7.24 seconds in the reused frontend, 9.80 seconds in decoder
prefill, and 129.48 seconds in decoder generation. Peak RSS was 537 MB and the
run did not swap.

## Remaining work

Decoder generation is now the largest measured phase and commonly uses only
two to three core equivalents. The next investigation should partition
independent output columns and attention heads while preserving each output's
reduction order. Persistent request-owned workers may then remove repeated
process and runtime initialization overhead.

The full recording has no complete reference transcript, so its result proves
trajectory identity with the merged CKE baseline and complete source
consumption, not an independent full-track WER target. The published
five-minute corpus retains the independent WER gate.
