# v8 Long-Context Prefill Sweep - 2026-07-08

## Purpose

Record the AVX2 v8 sweep that separated long-context prefill performance from
short-prompt practical generation quality.

This log is intentionally tracked because the raw benchmark artifacts live under
`build/` and are easy to lose. The results below should be referenced from the
PR/commit message for the AVX2 prefill work.

## Local Artifacts

- Markdown report:
  `build/reports/v8_practical_context_sweep_20260708-143018/report.md`
- JSON report:
  `build/reports/v8_practical_context_sweep_20260708-143018/sweep.json`
- Local HTTP view during the run:
  `http://10.0.0.32:8001/build/reports/v8_practical_context_sweep_20260708-143018/report.md`

These generated artifacts are not committed.

## Configuration

- Machine: Intel Core i7-14700T AVX2/AVX-VNNI host
- CK threads: `20`
- OpenMP threads: `1`
- Fixed-token lane:
  - Contexts: `1024`, `2048`, `4096`
  - Decode tokens: `64`
  - CK path: `ck-cli-v8 --prompt-tokens ... --no-chat-template --timing`
  - llama path: `llama-bench -p <ctx> -n 64 -t 20 -ngl 0`
- Practical prompt lane:
  - Context: `4096`
  - Max new tokens: `128`
  - CK only; llama-cli practical prompt lane was stopped because the first
    Gemma3 prompt did not terminate quickly enough for this sweep.
  - Important limitation: this was a short-prompt smoke. `--context 4096` only
    sets KV capacity; it does not create a 4096-token prompt.

## Practical Prompts

- `code_c_python_sql`: detailed C, Python, and SQL examples.
- `linked_list_c`: linked list in C with insert/delete/search/print.
- `database_hashmap`: relational database vs hash map explanation.
- `summary_text`: summarize a C-Kernel-Engine throughput paragraph.

## Fixed-Token Performance

### 2026-07-08 correction

The first fixed-token p4096 table from this sweep is **superseded**. It did not
reproduce after the runtime path corrections and should not be used as a PR
headline or public claim.

Fresh full recheck command:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python benchmarks/bench_v8_decoder_matrix.py \
  --models nanbeige4.1-3b-q4_k_m,qwen2-0.5b-q4_k_m,qwen3-0.6b-q8_0,qwen35-0.8b-q4_k_m,gemma3-270m-q5_k_m \
  --prompt 4096 \
  --decode 64 \
  --threads 20 \
  --repeats 1 \
  --json-out build/reports/v8_p4096_full_retest_20260708.json
```

Fresh p4096/n64 result:

| Model | Quant | CK prefill | llama prefill | CK/llama | CK decode | llama decode | CK/llama |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nanbeige 4.1 3B | Q4_K_M | 57.4 tok/s | 63.0 tok/s | 0.91x | 9.1 tok/s | 10.9 tok/s | 0.84x |
| Qwen2 0.5B | Q4_K_M | 210.8 tok/s | 223.3 tok/s | 0.94x | 44.3 tok/s | 48.2 tok/s | 0.92x |
| Qwen3 0.6B | Q8_0 | 144.3 tok/s | 173.6 tok/s | 0.83x | 26.4 tok/s | 34.1 tok/s | 0.77x |
| Qwen3.5 0.8B | Q4_K_M | 249.0 tok/s | 262.6 tok/s | 0.95x | 36.1 tok/s | 36.0 tok/s | 1.00x |
| Gemma3 270M | Q5_K_M | 365.0 tok/s | 761.3 tok/s | 0.48x | 29.1 tok/s | 74.7 tok/s | 0.39x |

Gemma3 was also rechecked alone and reached `540.3 tok/s` CK prefill vs
`1544.1 tok/s` llama prefill, showing run-to-run or thermal/cache variance but
still not reproducing the stale `31.5 tok/s` CK prefill row.

Superseded p4096/n64 table retained only as a cautionary artifact:

| Model | Quant | CK prefill | llama prefill | CK/llama | CK decode | llama decode | CK/llama |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nanbeige 4.1 3B | Q4_K_M | 230.3 tok/s | 63.1 tok/s | 3.65x | 12.1 tok/s | 11.1 tok/s | 1.09x |
| Qwen2 0.5B | Q4_K_M | 836.3 tok/s | 223.8 tok/s | 3.74x | 57.0 tok/s | 47.9 tok/s | 1.19x |
| Qwen3.5 0.8B | Q4_K_M | 921.4 tok/s | 258.9 tok/s | 3.56x | 39.9 tok/s | 32.4 tok/s | 1.23x |
| Gemma3 270M | Q5_K_M | 31.5 tok/s | 760.4 tok/s | 0.04x | 31.6 tok/s | 75.2 tok/s | 0.42x |
| Qwen3 0.6B | Q8_0 | 6075.0 tok/s | 174.0 tok/s | suspect | 45.3 tok/s | 34.0 tok/s | 1.33x |

## Practical CK Timings

Short real prompts do not exercise long batched prefill enough, so these values
should not be used as the long-prefill headline. `--context 4096` only sets KV
capacity; it does not make a 40-token prompt into a 4096-token prompt. These
rows are therefore short-prompt decode/coherence smokes, not proper
long-context practical tests.

| Model | Prompt token range | Prefill tok/s range | Decode tok/s range |
|---|---:|---:|---:|
| Nanbeige 4.1 3B | 40-88 | 28.0-43.6 | 11.4-12.8 |
| Qwen2 0.5B | 38-85 | 218.4-261.6 | 65.7-67.9 |
| Qwen3 0.6B | 38-85 | 213.9-270.5 | 53.7-54.6 |
| Qwen3.5 0.8B | 38-85 | 173.7-226.1 | 45.1-46.3 |
| Gemma3 270M | 40-86 | 30.0-31.5 | incomplete/30.3 |

Fresh 4096-context practical command shape:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python version/v8/scripts/ck_run_v8.py run <local-gguf> \
  --context-len 4096 \
  --prompt "Give me a detailed example of C, Python, and SQL code. Keep the answer concise but useful." \
  --max-tokens 512 \
  --temperature 0.0
```

Fresh 4096-context practical results:

| Model | Prompt tokens | Prefill | Decode cap/runs | Decode | Quality note |
|---|---:|---:|---:|---:|---|
| Nanbeige 4.1 3B Q4 | 53 | 46.85 tok/s | 512/512 | 13.54 tok/s | Coherent thinking, but 512 cap ended inside `<think>`; use higher reasoning budget/cap for final answer. |
| Qwen2 0.5B Q4 | 40 | 282.18 tok/s | 512/347 | 55.32 tok/s | Coherent enough, covers C/Python/SQL but SQL example is shallow. |
| Qwen3 0.6B Q8 | 29 | 198.53 tok/s | 512/350 | 39.16 tok/s | Coherent, exits thinking, gives concise examples. |
| Qwen3.5 0.8B Q4 | 29 | 40.43 tok/s | 512/512 | 36.84 tok/s | Coherent structure, but code quality is rough; calls C example C++. |
| Gemma3 270M Q5 | 31 | 29.95 tok/s | 512/135 | 30.92 tok/s | Coherent opening for size, but weak/repetitive C code. |

Artifacts:

- `build/reports/v8_p4096_full_retest_20260708.json`
- `build/reports/v8_context4096_practical_512_20260708/summary.md`
- `build/reports/v8_context4096_practical_512_20260708/summary.json`

Follow-up interactive Gemma3 run from the operator terminal was coherent for the
model size and did not match the stale low-prefill claim:

```text
Hello:
prompt eval: 58.29 ms / 13 tokens, 223.04 tok/s
decode: 319.76 ms / 10 runs, 31.27 tok/s

capital of France:
prompt eval: 62.58 ms / 17 tokens, 271.67 tok/s
decode: 256.17 ms / 8 runs, 31.23 tok/s

Shakespeare:
prompt eval: 60.45 ms / 16 tokens, 264.68 tok/s
decode: 1653.69 ms / 51 runs, 30.84 tok/s
```

## Output Quality Findings

- Nanbeige 4.1 3B: coherent thinking-mode output on the practical prompts.
- Qwen2 0.5B: coherent enough for the tested prompts.
- Qwen3 0.6B: initial practical sweep tail looked suspect after a plausible
  start, but a follow-up one-shot/interactive run produced coherent thinking
  and answer text. Treat Qwen3 as validation-pending, not a confirmed numerical
  regression.
- Qwen3.5 0.8B: coherent in the 4096-context 512-token run, but code quality is
  rough. It is a performance/correctness-quality follow-up, not a hard
  coherence failure.
- Gemma3 270M: follow-up interactive checks are coherent for model size, with
  short-prompt prefill around 220-285 tok/s and decode around 31-33 tok/s.
  Synthetic p4096 still lags llama.cpp, so Gemma remains a performance issue,
  not a coherence failure.

## Follow-Up Qwen3 / Qwen3.5 Checks

The runbook-style command without `--prompt` successfully converted, compiled,
and generated the visualizer, then exited with code `2` in this non-interactive
agent session because stdin is not a TTY:

```text
Error: interactive chat requires a TTY on stdin.
Run from a terminal or pass --prompt for one-shot generation.
```

Generated files:

- `/home/antshiv/.cache/ck-engine-v8/models/Qwen--Qwen3-0.6B-GGUF/libmodel.so`
- `/home/antshiv/.cache/ck-engine-v8/models/Qwen--Qwen3-0.6B-GGUF/weights.bump`
- `/home/antshiv/.cache/ck-engine-v8/models/Qwen--Qwen3-0.6B-GGUF/ir_report.html`

Direct one-shot timing from the compiled runtime:

```text
prompt: Give me a detailed example of C, Python, and SQL code. Explain what each example does.
prefill 28 tok: 131.5 ms, 213.0 tok/s
decode 127 tok: 2450.6 ms, 51.8 tok/s, 19.3 ms/tok
```

User interactive terminal run with a longer cap was also coherent:

```text
prompt eval: 126.41 ms / 23 tokens, 181.95 tok/s
decode: 11560.50 ms / 512 runs, 44.29 tok/s
reasoning: started=1 ended=1 tokens=390
stop: max_tokens reached
```

Conclusion: Qwen3 Q8 speed should still be checked for token accounting and
logit parity, but current evidence does not show a critical coherence
regression.

The user also ran Qwen3 interactively with a `Hello` prompt:

```text
prompt eval: 74.23 ms / 10 tokens, 134.72 tok/s
decode: 1716.11 ms / 90 runs, 52.44 tok/s
reasoning: started=1 ended=1 tokens=77
stop: eos token 151645
```

Qwen3.5 was retested through `cks-v8-run` after a fresh compile under the
canonical cache directory:

```text
/home/antshiv/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF/libmodel.so
```

Observed Qwen3.5 timing:

```text
Hello prompt:
prompt eval: 222.59 ms / 10 tokens, 44.93 tok/s
decode: 332.69 ms / 15 runs, 45.09 tok/s

code prompt:
prompt eval: 517.15 ms / 24 tokens, 46.41 tok/s
decode: 11412.71 ms / 512 runs, 44.86 tok/s
reasoning: started=1 ended=1 tokens=3
stop: max_tokens reached
```

Qwen3.5 output was coherent enough for a practical smoke. The earlier
Qwen3/Qwen3.5 garbage tails are now attributed to an invalid benchmark setup:
stale compiled runtime paths and too-short practical prompts, not a confirmed
model/kernel regression.

Harness correction:

- `benchmarks/bench_v8_decoder_matrix.py` now points Qwen3.5 to
  `unsloth--Qwen3.5-0.8B-GGUF`, matching `cks-v8-run`.
- `benchmarks/compare_ck_llama_v8.py` now points Qwen3 to
  `Qwen--Qwen3-0.6B-GGUF`, matching `cks-v8-run`.

## Corrected Long-Prompt Attempt

A follow-up Qwen3 test attempted to use actual long text prompts with 512-token
decode caps. The 512-token and 1024-token prompt rows are valid; the 4096
context row is invalid because the generated prompt collapsed to only 6 tokens
as reported by CK timing.

```text
ctx=1024: prompt=512 tokens, prefill=247.5 tok/s, decode=511 tokens at 35.0 tok/s
ctx=2048: prompt=1024 tokens, prefill=207.8 tok/s, decode=511 tokens at 54.4 tok/s
ctx=4096: invalid, CK reported prompt=6 tokens
```

Artifact:

- `build/reports/qwen3_corrected_practical_long_context_20260708.json`

This confirms the next practical sweep must generate prompts by tokenizer count,
not approximate word count.

## What Changed / What The Sweep Proves

This sweep did not prove that every v8 model is fixed. It proved a narrower,
useful point:

1. The stale p4096 table cannot be used as evidence that CK beats llama.cpp.
   Fresh full recheck puts CK near llama.cpp for Nanbeige, Qwen2, and Qwen3.5
   prefill, but still behind; Gemma3 and Qwen3 Q8 are farther behind.
2. The practical interactive signal is still good: Qwen3, Qwen3.5, and Gemma3
   produce coherent output in fresh `cks-v8-run` terminal runs, and short-prompt
   speed is real.
3. Short practical prompts are dominated by overhead and decode. They do not
   show the same prefill speedup because they only contain about 40-90 prompt
   tokens.
4. Qwen3/Qwen3.5 coherence is not a confirmed regression after fresh
   `cks-v8-run` checks, but their benchmark numbers must be regenerated with
   canonical runtime paths and token-counted prompts.
5. Gemma3 is a separate performance/routing problem, not evidence that the Q4
   long-prefill path failed.

## Likely Issue Split

- Q4 long prefill: improved enough to justify continued roofline work, but the
  old "CK beats llama.cpp by 3.6x" table is invalid until regenerated and
  reproduced.
- Qwen3/Qwen3.5: validation-pending for performance accounting, but no longer
  considered confirmed coherence regressions after fresh `cks-v8-run` checks.
- Benchmark harness: stale runtime cache paths can produce false coherence
  failures. Always align benchmark `run` directories with the directories that
  `cks-v8-run --force-convert --force-compile` writes.
- Gemma3: coherent in interactive use, but still slower than llama.cpp in the
  fresh synthetic p4096 recheck.
- Qwen3 Q8 p4096 speed: suspect until output quality and token accounting are
  verified.

## Next Steps

1. Rerun CK-vs-llama fixed-token and practical-prompt sweeps after the runtime
   path corrections.
2. Replace the flawed practical lane with actual token-counted long prompts:
   - context 1024: about 512 prompt tokens, 512 decode cap
   - context 2048: about 1024-1536 prompt tokens, 512 decode cap
   - context 4096: about 3072-3584 prompt tokens, 512 decode cap
3. Verify Qwen3 p4096 token accounting, because the fixed-token throughput
   number is unusually high relative to the rest of the model family.
4. If output diverges, run first-token and multi-token logit parity with the
   same prompt and generated prefix.
5. Attribute the first Qwen3/Qwen3.5 divergence by layer/op before changing
   speed kernels further.
6. Inspect Gemma3 v8 lowering and kernel-map routing. Confirm whether Gemma3 is
   using the intended batched/sliding prefill path.
7. After correctness is stable, rerun Advisor/VTune on:
   - Nanbeige p4096 prefill
   - Qwen2 p4096 prefill
   - Qwen3.5 p4096 only after coherence/parity is fixed
   - Gemma3 after routing is fixed
