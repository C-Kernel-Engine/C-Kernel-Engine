# Gemma3 Q5_1 Prefill Tuning - 2026-07-08

## Purpose

Record the Gemma3-specific v8 AVX2 tuning pass. This log is tracked so the
benchmark trail is tied to the code change instead of only living in `build/`
artifacts.

Gemma3 270M Q5_K_M was lagging llama.cpp badly in the fixed-token p4096 lane.
The v8 profile showed the real prefill bottleneck was not generic Q5_K alone:
Gemma3 uses the Q5_1 x Q8_1 contract path heavily for q/k projections and
especially MLP gate/up.

## Change Summary

- Added an AVX2/VNNI Q5_1 x Q8_1 dot path using `vpdpbusd` when the compiler
  target exposes `__AVXVNNI__`.
- Replaced scalar high-bit reconstruction with a 4-bit lookup table that expands
  Q5_1 high bits into byte lanes.
- Reused each quantized activation row across eight output columns in
  `gemm_nt_q5_1_q8_1`.
- Fixed the fixed-token benchmark harness so CK p4096 runs use the requested
  token count:
  - explicit `--context`
  - `--no-chat-template`
  - `--no-stream`
  - default context = `prompt + decode + 8`

## Machine And Configuration

- Host: Intel Core i7-14700T AVX2/AVX-VNNI
- CK threads: `20`
- OpenMP threads: `1`
- Model: `unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf`
- Fixed-token lane: `p4096/n64`, CK context `4168`
- Local generated artifacts:
  - `build/reports/gemma3_q5_prefill_profile_p4096_20260708.json`
  - `build/reports/gemma3_q5_prefill_profile_p4096_after_q51_lut_20260708.json`
  - `build/reports/gemma3_q5_p4096n64_final_20260708.json`

## Prefill Profile Delta

Profile command family:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python benchmarks/profile_v8_prefill_ops.py \
  --models gemma3-270m-q5_k_m \
  --prompt 4096 \
  --decode 1 \
  --threads 20
```

| Metric | Before | After |
|---|---:|---:|
| CK profiled prefill tok/s | 569.5 | 874.4 |
| Prompt wall time | 7192.4 ms | 4684.2 ms |
| Profiled op coverage | 97.5% | 96.9% |
| `mlp_gate_up` Q5_1 | 3130.0 ms | 1346.2 ms |
| `q_proj` Q5_1 | 834.6 ms | 340.5 ms |
| `k_proj` Q5_1 | 217.3 ms | 92.9 ms |
| Q5_1 `v_proj` layers | 108.2 ms | 45.2 ms |

After the Q5_1 fix, the next prefill hotspots are no longer just Q5_1:

| Op/kernel | Time | Share |
|---|---:|---:|
| `gemm_nt_q5_1_q8_1` / `mlp_gate_up` | 1346.2 ms | 29.7% |
| `attention_forward_causal_head_major_gqa_flash_strided_sliding` | 911.6 ms | 20.1% |
| `geglu_forward_exact` | 710.2 ms | 15.6% |
| `gemm_nt_q8_0_q8_0_contract` / `v_proj` | 422.1 ms | 9.3% |
| `gemm_nt_q5_1_q8_1` / `q_proj` | 340.5 ms | 7.5% |
| `gemm_nt_q5_k` / `mlp_down` | 192.5 ms | 4.2% |
| `gemm_nt_q6_k_q8_k` / `mlp_down` | 105.0 ms | 2.3% |

## Corrected CK-vs-llama Fixed-Token Result

The earlier Gemma3 p4096 rows were misleading because the CK harness could clip
the requested fixed-token prompt. The corrected harness reports all 4096 prompt
tokens:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python benchmarks/bench_v8_decoder_matrix.py \
  --models gemma3-270m-q5_k_m \
  --prompt 4096 \
  --decode 64 \
  --context 4168 \
  --threads 20 \
  --repeats 1 \
  --timeout 1200 \
  --json-out build/reports/gemma3_q5_p4096n64_final_20260708.json
```

| Model | Quant | llama prefill | CK prefill | CK/llama | llama decode | CK decode | CK/llama |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma3 270M | Q5_K_M | 1549.2 tok/s | 859.3 tok/s | 0.55x | 118.3 tok/s | 30.5 tok/s | 0.26x |

Interpretation:

- The Q5_1 prefill kernel itself improved materially.
- Gemma3 prefill is still behind llama.cpp end-to-end.
- Decode is still the largest relative Gemma3 gap.
- The next Gemma3 work should target sliding attention, GEGLU/fusion, Q8_0
  projection routing, and decode GEMV paths.

## Correctness And Smoke Checks

Build and kernel parity:

```bash
make -B build/libckernel_engine.so
objdump -d -Mintel build/libckernel_engine.so | rg "vpdpbusd"
.venv/bin/python -m py_compile \
  benchmarks/bench_v8_decoder_matrix.py \
  benchmarks/compare_ck_llama_v8.py
make test-kernels
```

Results:

- `vpdpbusd` is present in `build/libckernel_engine.so`.
- `py_compile` passed.
- `make test-kernels` passed: `58/58`.
- Focused Q5_1 x Q8_1 randomized scalar-contract parity passed:

```text
shape M=1 N=7 K=32: max=1.19209e-07 mean=4.0446e-08
shape M=3 N=17 K=64: max=1.90735e-06 mean=2.41925e-07
shape M=5 N=33 K=160: max=1.90735e-06 mean=3.63001e-07
shape M=9 N=64 K=640: max=2.28882e-05 mean=2.07134e-06
Q5_1 x Q8_1 focused parity OK
```

Practical Gemma3 smoke:

```text
Prompt: What is the capital of France? Give one sentence.
Response: Paris
Please provide the city name.

prompt eval: 61.24 ms / 21 tokens, 342.89 tok/s
decode: 264.89 ms / 8 runs, 30.20 tok/s
stop: eos token 106
```

This is coherent enough for a 270M model and does not indicate a kernel-level
coherence regression.

## Next Steps

1. Fix Gemma3 decode: profile `gemv_q5_1_q8_1`, final logits, and Q5_K/Q6_K
   decode paths.
2. Improve remaining Gemma3 prefill hotspots: sliding attention, GEGLU, Q8_0
   contract projections, and MLP down.
3. Add a direct Q5_1 parity test to the committed kernel test suite instead of
   keeping it only as an ad hoc ctypes check.
4. Move to Gemma4 only after Gemma3 decode and prefill are both closer to
   llama.cpp.
