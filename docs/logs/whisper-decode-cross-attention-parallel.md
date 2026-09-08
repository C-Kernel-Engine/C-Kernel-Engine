# Whisper decode cross-attention head parallelism

## Scope

Whisper decoder cross-attention has one query token and independent attention
heads. The existing provider partitions query rows, so decode uses one worker.
This change adds a decode-only provider that assigns complete heads to CKE's
persistent thread pool. Each worker invokes the established one-head FP32
attention path; QK, softmax, and probability-value reduction order remain
unchanged.

Prefill retains the existing compact `[Tq,Tk]` scratch contract. Decode uses a
separate `[H,Tk]` scratch contract so every worker owns its score row. The
provider is selected through the decoder circuit's numerical contract rather
than a model-name branch.

## P3 measurement

Host: Intel Core i7-14700T P3. Model: `openai/whisper-base` with FP32-preserved
encoder and decoder. Input: retained five-minute, 16 kHz mono nightly fixture.
CKE used 20 threads and OpenMP used one thread. Conversion and compilation were
excluded from the measured inference runs.

| Measurement | Output-parallel baseline | Cross-attention median of 3 |
|---|---:|---:|
| Decoder generation | 9.760 s | 6.458 s |
| Decoder prefill | 1.111 s | 1.140 s |
| Encoder | 10.443 s | 10.650 s |
| Frontend | 0.848 s | 0.844 s |

The three decoder-generation measurements were 6.450, 6.458, and 6.685
seconds, a 33.8% median reduction. One complete measured run took 23.49 seconds
wall time and averaged 699% process CPU. This is evidence for Whisper-base on
this P3, not a general claim for every model or CPU.

## Complete-track measurement

The retained 42:22 recording completed in 3:22.40 wall time. Decoder generation
fell from 99.556 seconds with output-projection parallelism alone to 65.673
seconds with cross-attention head parallelism, a 34.0% reduction. The process
averaged 739% CPU and reached 525 MiB maximum resident memory. There were no
major page faults or swaps.

| Measurement | Output-parallel baseline | Cross-attention |
|---|---:|---:|
| Decoder generation | 99.556 s | 65.673 s |
| Decoder prefill | 9.119 s | 9.307 s |
| Encoder | 88.681 s | 89.071 s |
| Frontend | 7.204 s | 7.148 s |

The candidate matched all 9,870 generated token IDs, the complete transcript,
all 92 source-window boundaries, and every per-window feature and encoder hash.
The report records schema version 4 provenance, including both engine hashes
and the requested 256-token per-window limit.

## Correctness

- All three runs matched all 974 generated token IDs.
- Transcript text and all 11 source-window boundaries matched.
- Every per-window feature and encoder-output SHA-256 matched.
- An 8,192-dispatch poison replay matched the serial provider bit exactly.
- The numerical execution suite passed 69 tests and 98 subtests.
- The audio contract suite passed; its optional environment-gated real PyTorch
  E2E lane reported an explicit skip.

The comparison used `--max-tokens 256`. A diagnostic run that omitted this
setting hit the current 128-token default in one window and changed subsequent
window boundaries. That was a request-policy mismatch, not a numerical
regression.

## Provenance hardening

Whisper E2E schema version 4 records both generated `libmodel.so` hashes and
shared `libckernel_engine.so` hashes, plus the requested per-window token limit.
Previously, materially different shared runtimes or token limits could appear
equivalent in retained evidence.

Remaining utilization work includes persistent per-request workers and encoder
GELU/GEMM improvements. Self-attention head dispatch was investigated but not
changed: Whisper-base's 448-token cache was below the measured useful dispatch
threshold, and routing its serial fallback through a callback changed the
compiled numerical schedule.
