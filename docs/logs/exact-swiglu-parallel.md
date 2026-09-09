# Exact SwiGLU row parallelism

Date: 2026-09-08

## Scope

This change parallelizes independent token rows in the FP32 exact and GGML
SwiGLU providers. It does not change the sigmoid approximation, per-output
arithmetic, BF16 providers, split-input providers, backward kernels, or the
opt-in fast path.

Small calls stay serial. Calls with separate input and output buffers may
partition rows directly. The generated runtime commonly compacts `[T, 2D]`
input to `[T, D]` output in place, so those calls execute dependency-safe row
waves. Arbitrary row order is incorrect because output row `t` overlaps input
row `floor(t / 2)`.

## Numerical evidence

- P3 AVX2: serial and four-thread output was bit-exact for both providers at
  `1x127`, `2x16384`, `17x4097`, and `128x4096`, with separate and in-place
  buffers. The in-place cases repeat eight times.
- Ryzen AVX-512: the same kernel test passed.
- Ryzen Qwen3 0.6B Q8_0: seven matched-parent 1,024-token runs produced the
  same first-logit hash (`4974e1cc71993d43`) and generated IDs
  (`26047, 2075`) in baseline and candidate runtimes.
- P3 v7 Nanbeige 4.1 3B: candidate and untouched parent both build and run,
  produce the same response, and reproduce the existing coherence-parser and
  first-token-parity failures. The v8 fast family regression passes.

The initial unrestricted row implementation failed this full-model gate: it
produced a different, nondeterministic first-logit hash at every multi-threaded
run. The in-place regression test preserves that failure mode.

## Ryzen performance evidence

Hardware: Ryzen 9 9950X3D, CPUs 0-15, 16 workers, Qwen3 0.6B Q8_0, identical
1,024-token input, parent commit `4a433162b53d8fbeb1c4aefe04b297235396172a`.

| Runtime | Prefill samples (ms) | Median |
|---|---|---:|
| Parent | 864.3, 848.8, 857.9, 882.1, 870.7, 848.7, 943.9 | 864.3 ms |
| Candidate | 731.9, 746.9, 745.8, 743.0, 809.3, 932.4, 1365.9 | 746.9 ms |

The median prefill improvement is 15.7%. Raw artifacts are retained under
`/data/cke/profiles/exact-swiglu-20260908` on the measured Ryzen host.

## Remaining work

The GGML microkernel scales well at width 4096 but saturates near 1.9 ms for
`1024x11008` on this Ryzen. That boundary needs hardware-counter analysis; it
is not evidence for changing numerical semantics or globally increasing the
thread count. This change also does not claim gains for model paths selecting
other activation providers.
