# Suggested Commit Message

```text
perf(v8): speed up Gemma3 Q5_1 prefill

Why: Gemma3 270M Q5_K_M prefill was dominated by the Q5_1 x Q8_1 contract path,
especially MLP gate/up and q/k projections, and the fixed-token benchmark
harness could clip CK p4096 runs below the requested prompt length.
Test: make -B build/libckernel_engine.so; objdump confirmed vpdpbusd in the
rebuilt shared library; py_compile benchmark scripts; make test-kernels; focused
Q5_1 x Q8_1 scalar-contract parity; corrected Gemma3 p4096/n64 CK-vs-llama
benchmark; Gemma3 practical coherence smoke.

Changes:

- Add an AVX2/VNNI Q5_1 x Q8_1 dot path using vpdpbusd when __AVXVNNI__ is
  available.
- Replace scalar Q5_1 high-bit reconstruction with a nibble lookup table.
- Reuse each quantized activation row across eight output columns in
  gemm_nt_q5_1_q8_1.
- Fix the fixed-token benchmark harness to pass explicit context, disable chat
  templates, and disable streaming for deterministic token accounting.

Gemma3 p4096 profile delta:

| Metric | Before | After |
|---|---:|---:|
| CK profiled prefill | 569.5 tok/s | 874.4 tok/s |
| Prompt wall time | 7192.4 ms | 4684.2 ms |
| mlp_gate_up Q5_1 | 3130.0 ms | 1346.2 ms |
| q_proj Q5_1 | 834.6 ms | 340.5 ms |
| k_proj Q5_1 | 217.3 ms | 92.9 ms |

Corrected fixed-token p4096/n64 CK-vs-llama:

| Model | llama prefill | CK prefill | CK/llama | llama decode | CK decode |
|---|---:|---:|---:|---:|---:|
| Gemma3 270M Q5_K_M | 1549.2 tok/s | 859.3 tok/s | 0.55x | 118.3 tok/s | 30.5 tok/s |

Benchmark log:

- logs/2026-07-08-gemma3-q51-prefill/README.md

Next:

- Gemma3 decode remains the largest relative gap.
- Remaining prefill hotspots are sliding attention, GEGLU/fusion, Q8_0 contract
  projection routing, and MLP down.
- Add committed Q5_1 focused parity coverage.
```
