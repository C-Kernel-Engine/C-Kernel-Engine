# Whisper persistent-worker validation

Date: 2026-09-09

## Scope

The long-audio runner previously launched and initialized one encoder process
and one decoder process for every audio window. The persistent lifecycle keeps
one isolated process for each generated runtime while preserving the existing
NPY encoder-memory transport.

Each decoder window begins with `ck_model_kv_cache_reset()`. Binding the next
encoder output through `ck_model_set_encoder_memory()` invalidates the generated
cross-attention K/V cache. Immutable weights, prepared constants, the tokenizer,
thread pools, and the full-feature memory map remain resident.

Use `--worker-lifecycle per-window` to reproduce the old lifecycle for diagnosis.
Persistent workers are the default.

Shutdown is part of the request contract. A worker acknowledges `closed` only
after `ck_model_free()` completes, and the parent validates the response role,
response deadline, process termination, and zero exit status. A worker crash,
timeout, forced termination, or teardown exception therefore cannot be reported
as a successful lifecycle. If inference and cleanup both fail, the inference
exception remains primary and includes the cleanup failure as additional
diagnostic evidence. If inference succeeds but cleanup fails, the completed
transcript and JSON report remain available with `status: error` before the
command returns failure.

## Numerical and task evidence

The A/B comparison used the same Whisper-base generated runtimes, 16 kHz mono
input, English transcription, timestamp decoding, and greedy generation with a
256-token limit per window.

- Input duration: 42:22
- Windows: 92
- Generated tokens: 9,870
- Exact token trajectories: 92/92
- Exact feature hashes: 92/92
- Exact encoder-output hashes: 92/92
- Exact window boundaries and timestamp events: 92/92
- Exact stop reasons and transcript text: 92/92

Runtime identities:

- Encoder `libmodel.so`: `861840ed7fceef11e8e0000f44daae0b4c6171a869874ff344afedb1562d9d4b`
- Decoder `libmodel.so`: `b5aeeb3ce5073972cd9711795a1d67e807af5863ed691e353e0ce0efecb5f9ee`

## P3 performance

Host: Intel Core i7-14700T, 20 cores / 28 logical CPUs. Both runs used the same
runtime artifacts and input on an otherwise comparable local system.

| Metric | Per-window processes | Persistent workers |
|---|---:|---:|
| Wall time | 171.60 s | 138.73 s |
| Encoder processes / model initializations | 92 | 1 |
| Decoder processes / model initializations | 92 | 1 |
| Frontend | 7.14 s | 7.04 s |
| Encoder | 59.00 s | 58.40 s |
| Decoder prefill | 7.57 s | 8.02 s |
| Decode | 65.14 s | 63.13 s |
| Outside phase timers | 32.75 s | 2.14 s |
| Peak RSS | 526 MiB | 525 MiB |
| Minor page faults | 2,269,756 | 256,730 |

The measured wall-time improvement is 19.2%. This is orchestration and model
lifecycle work, not a claim that encoder or decoder arithmetic became faster.

## Ryzen cross-hardware check

The same five-minute artifact and copied runtime binaries were replayed on the
AMD Ryzen 9 9950X3D host with 16 physical cores / 32 logical CPUs. The per-window
control completed in 11.708 s and persistent execution completed in 9.490 s, an
18.9% wall-time improvement. All 1,126 generated tokens, feature and encoder
hashes, window boundaries, timestamp events, and stop reasons matched exactly.
Outside-phase time fell from 2.72 s to 0.44 s.

The copied runtime is the same AVX2 artifact used on P3, so this result isolates
worker lifecycle across hardware; it is not a native AVX-512 Ryzen benchmark.

Retained evidence is under:

`/data/cke/workloads/whisper/kernel2-youtube-video/persistent-worker-ab/`

## Validation

- `39 passed, 2 skipped` in `tests/test_v8_whisper_runner.py`; the skipped cases
  require separately configured real-model fixtures.
- The focused runner tests include initialization failure, teardown failure,
  process crash, response timeout with forced cleanup, exit-status validation,
  preservation of a primary inference exception, and retention of completed
  output after cleanup failure. A real subprocess also acknowledges shutdown,
  ignores `SIGTERM`, and hangs; the parent must escalate to `SIGKILL`, reap it,
  and preserve the failure for repeated `close()` calls.
- A real five-minute P3 replay completed with exit status zero after both
  generated runtimes acknowledged post-free shutdown. Its report recorded
  `status: ok`, 11 windows, 974 tokens, and explicit encoder/decoder `closed`
  responses.
- 31 audio/encoder/Cohere Transcribe contract tests passed.
- Two Whisper conversion tests passed.
- 13 long-audio and benchmark contract tests passed.
- Five-minute P3 A/B/A controls were also exact across 1,126 tokens and 12
  windows; persistent execution completed in 15.80 s versus 19.76 s and 19.79 s
  controls. The Ryzen five-minute A/B was exact and improved 18.9%.
