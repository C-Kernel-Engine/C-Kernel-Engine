# Suggested Commit Message

```text
perf(v8): record AVX2 long-context prefill sweep

Add a tracked benchmark worklog for the 2026-07-08 v8 AVX2 sweep.
Align the Qwen benchmark runtime paths with `cks-v8-run` canonical cache
directories so future sweeps do not accidentally use stale compiled runtimes.

The sweep separates fixed-token long-prefill throughput from practical
prompt coherence:

Fresh fixed-token p4096/n64 CK-vs-llama retest:

| Model | CK prefill | llama prefill | CK/llama | CK decode | llama decode |
|---|---:|---:|---:|---:|---:|
| Nanbeige 4.1 3B Q4 | 57.4 tok/s | 63.0 tok/s | 0.91x | 9.1 | 10.9 |
| Qwen2 0.5B Q4 | 210.8 tok/s | 223.3 tok/s | 0.94x | 44.3 | 48.2 |
| Qwen3 0.6B Q8 | 144.3 tok/s | 173.6 tok/s | 0.83x | 26.4 | 34.1 |
| Qwen3.5 0.8B Q4 | 249.0 tok/s | 262.6 tok/s | 0.95x | 36.1 | 36.0 |
| Gemma3 270M Q5 | 365.0 tok/s | 761.3 tok/s | 0.48x | 29.1 | 74.7 |

Fresh 4096-context practical prompt smoke with 512-token cap:

| Model | Prompt tok | Prefill | Decode | Quality note |
|---|---:|---:|---:|---|
| Nanbeige 4.1 3B Q4 | 53 | 46.85 tok/s | 13.54 tok/s | Coherent thinking, cap ended inside thought |
| Qwen2 0.5B Q4 | 40 | 282.18 tok/s | 55.32 tok/s | Coherent enough, shallow SQL |
| Qwen3 0.6B Q8 | 29 | 198.53 tok/s | 39.16 tok/s | Coherent, exits thinking |
| Qwen3.5 0.8B Q4 | 29 | 40.43 tok/s | 36.84 tok/s | Coherent structure, rough code |
| Gemma3 270M Q5 | 31 | 29.95 tok/s | 30.92 tok/s | Coherent opening for 270M, repetitive C |

The earlier p4096 table claiming 3.5x+ CK prefill wins did not reproduce and is
kept only as a superseded cautionary artifact in the log.

Validity notes:

- `--context-len 4096` sets capacity; it is not a 4096-token prompt unless the
  prompt is tokenizer-counted to that length.
- Practical 4096-context prompt tokens were still short (29-53 tokens), so those
  rows validate runtime health and coherence, not long-prefill saturation.
- Qwen3/Qwen3.5 are not confirmed coherence regressions after fresh terminal
  runs.
- Gemma3 is coherent for its size but remains behind llama.cpp in synthetic
  p4096 and produces weak code.

Benchmark log:

- logs/2026-07-08-v8-long-context-prefill-sweep/README.md

Generated local artifacts, not committed:

- build/reports/v8_p4096_full_retest_20260708.json
- build/reports/v8_context4096_practical_512_20260708/summary.md
- build/reports/v8_context4096_practical_512_20260708/summary.json
```

## Suggested PR Notes

```text
This PR includes a tracked benchmark log for the AVX2 v8 long-context sweep:

logs/2026-07-08-v8-long-context-prefill-sweep/README.md

Main findings:

- The original fixed-token table was stale/suspect and is now explicitly
  superseded.
- Fresh p4096/n64 puts CK close to llama.cpp on Nanbeige, Qwen2, and Qwen3.5
  prefill, but not ahead.
- Fresh 4096-context practical prompts run successfully across Nanbeige, Qwen2,
  Qwen3, Qwen3.5, and Gemma3.
- Coherence is reasonable for Qwen2/Qwen3 and present for Qwen3.5/Gemma3, but
  code quality still needs model-specific evaluation.

Do not treat this as a long-context practical benchmark yet: the practical
prompt was only 29-53 tokens depending on tokenizer/template. The next lane
needs tokenizer-counted prompts near 1024/2048/4096 tokens.

Next steps after this PR:

1. Generate token-counted practical prompts for 1024/2048/4096 context tests.
2. Add these prompt lanes to the pre-push/nightly performance script.
3. Profile p4096 Nanbeige/Qwen2/Qwen3.5 with VTune/Advisor to close the last
   5-17% prefill gap vs llama.cpp.
4. Investigate Qwen3 Q8 and Gemma3 gaps separately.
5. Keep first-token/multi-token parity gates active for any kernel speed patch.
```
