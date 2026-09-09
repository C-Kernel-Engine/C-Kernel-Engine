# Whisper long-audio frontend reuse

## Scope

This change measures the generated Whisper frontend and encoder separately, then
reuses one request-scoped full-recording log-Mel tensor across timestamp-selected
encoder windows. The full tensor is stored in the runner's temporary directory;
there is no process-global cache or cross-request state.

The generated full-feature entry point is assembled from the resolved
`audio_feature_window` call ABI. It keeps the existing PCM decode, STFT, mel,
global-maximum normalization, FP32 storage, and reduction order. Individual
encoder workers copy a hop-aligned slice into the existing fixed-size
`audio_features` activation and zero-pad the tail.

## P3 measurement

Host: Intel Core i7-14700T P3, default CKE topology. Model:
`openai/whisper-base`, FP32-preserved encoder and decoder. Input: retained
five-minute, 16 kHz mono nightly fixture. Builds and conversion were excluded.

| Measurement | Repeated frontend | Reused frontend (median of 3) |
|---|---:|---:|
| Wall time | 37.26 s | 29.04 s |
| Frontend time | 9.161 s | 0.864 s |
| Encoder time | 10.527 s | 10.446 s |
| Decoder prefill | 1.008 s | 1.096 s |
| Decoder generation | 12.602 s | 12.673 s |
| Windows | 11 | 11 |

The three reused-frontend wall measurements were 29.04, 28.91, and 29.35
seconds. The 22.1% median wall improvement is evidence for the removed repeated
work, not a general claim for every Whisper model and CPU. The retained
repeated-frontend baseline and a fresh split-timing replay measured 37.23 and
37.26 seconds respectively.

## Correctness

- All 974 generated token IDs matched the retained baseline.
- All source window starts and consumed ends matched.
- Every per-window feature SHA-256 matched.
- Every per-window encoder-output SHA-256 matched.
- Long-audio certification passed at 5.65% WER with 125 timestamps and full
  300-second coverage.
- `make test-audio-v8-contracts` passed, including 23 runner tests, 6
  long-audio certification tests, and the existing generated audio contracts.

## Boundaries

Reuse is enabled only for long PCM WAV inputs whose source rate already matches
the model rate. Short audio and inputs whose WAV geometry cannot be read by the
standard parser retain the existing generated frontend path. Resampled
long-audio reuse remains open because globally phased resampling needs its own
explicit numerical contract.

Remaining utilization work is separate: parallel exact FP64-erf GELU,
single-token GEMM output partitioning, attention head ownership, and persistent
encoder/decoder workers all require their own parity and performance evidence.
