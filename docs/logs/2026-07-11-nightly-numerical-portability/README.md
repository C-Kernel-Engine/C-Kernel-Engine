# Nightly numerical portability repair

## Failure cluster

The 2026-07-11 nightly reported four failed rows. One was an aggregate duplicate,
leaving three leaf failures:

- RoPE cache precomputation
- fused attention decode
- FP16 split-KV attention against llama.cpp

No tolerance was widened.

## Root causes

### RoPE cache

The FP32 cache contract was computed through host `long double` `logl`/`expl`.
The final float therefore depended on the runner's libm and long-double ABI.
Computing the frequency directly with FP32 `powf` reduced the focused cache
maximum error to `5.96e-8` on the AVX2 node.

### Fused decode

The test compared a fused CK circuit directly with a PyTorch BLAS result. The
BLAS reduction changed with runner CPU dispatch and sat on the old `5e-3`
boundary. Fusion equivalence now compares against the unfused CK circuit with
the same deterministic attention contract. PyTorch remains a reported
diagnostic. The attention-only fusion is bit-exact and the fused-MLP error is
`4.27e-4` under a tightened `1e-3` bound.

### FP16 split-KV

CK emulated llama.cpp's AVX2 FP16 dot reduction on every x86 ISA. An AVX-512
llama.cpp build uses a different lane width and reduction tree. CK now has
matching AVX2, AVX-512 FP16-to-FP32, and native AVX-512-FP16 trees. AVX-512 and
AVX-512-FP16 compile-only checks passed locally; native execution remains part
of the Xeon/nightly lane.

## Validation

- Nightly kernels category: 32/32 pass
- Full attention: 7/7 pass
- Numerical contract aggregate: pass, including split-KV 14/14
- RoPE cache: `5.96e-8` max error
- Fused attention-only decode: exact
- Fused MLP decode: `4.27e-4` max error, `1e-3` bound
- AVX-512 and AVX-512-FP16 source builds: pass with GCC and ICX
- Missing local llama parity artifacts: consistently reported as skipped

GitHub's AVX-512 execution remains the authoritative confirmation for the new
AVX-512 reduction branch.
