ADR-0004: Deterministic Layout + Canaries for Debug
===================================================

Status
- Accepted

Context
- IR v2 used dynamic allocation, making memory bugs non-reproducible.
- We need deterministic layout for debugging and visualization tools.
- Buffer overflows in GEMM/attention are common and hard to diagnose.

Decision
- All tensor offsets computed at codegen time, baked into generated C code.
- 64-byte alignment for AVX-512 cache line optimization.
- Insert 64-byte canary markers (0xDEADBEEF pattern) between tensors in debug builds.
- Export memory_layout.json with all offsets for tooling.

Layout Structure:
```
┌──────────────────────────────────────────────┐
│ Header (64B): magic, version, canary offsets │
├──────────────────────────────────────────────┤
│ Weights (read-only, shared)                  │
├──────────────────────────────────────────────┤
│ CANARY_0 (64B debug only)                    │
├──────────────────────────────────────────────┤
│ Activations (per-mode buffers)               │
├──────────────────────────────────────────────┤
│ CANARY_1 (64B debug only)                    │
├──────────────────────────────────────────────┤
│ KV Cache                                     │
└──────────────────────────────────────────────┘
```

Consequences
- Reproducible: same offsets every run, bugs are reproducible.
- Early detection: canary violations caught before corruption spreads.
- No malloc overhead: single allocation, no fragmentation.
- Canaries disabled in release builds (zero overhead).

Alternatives
- Runtime allocation per tensor (rejected: non-reproducible, fragmentation).
- No canaries (rejected: loses valuable debug capability).
