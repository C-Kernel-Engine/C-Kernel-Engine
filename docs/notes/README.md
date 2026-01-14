# C Kernel Engine

![C Kernel Engine cover](assets/cover_image.png)

A C-first kernel library for LLMs and deep-learning workloads.

> Cover image generated with Google Gemini (banana-themed concept art).

This project focuses on a **small, high-impact set of kernel shapes** that appear in transformer-style models, instead of trying to cover every BLAS case. The goal is to:

- Implement a clean kernel DSL around a handful of GEMM and elementwise patterns.
- Tune those kernels aggressively for CPU (x86/ARM/RISC-V, AVX2/AVX-512, etc.).
- Keep the implementation understandable and hackable for systems and HPC engineers.

See `docs/01-llm-kernel-shapes.md` for the core math that drives the design.

**Start here:** `docs/00-quickstart.md` (build, tests, and codegen).

The code generator (`build/ck_ir_demo`) can emit a `libmodel.so` that supports:
- Prompt **prefill** (full forward)
- Autoregressive **decode** with KV cache (inference-only)
- **Backward** / training (teacher forcing full forward+backward)
