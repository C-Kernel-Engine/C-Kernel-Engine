#!/usr/bin/env python3
"""Generate the README capability map for C-Kernel-Engine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "cke-capability-map.png"


def rounded_box(ax, xy, wh, title, body, fc, ec="#0f172a", tc="#f8fafc"):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
        alpha=0.98,
    )
    ax.add_patch(box)
    ax.text(x + 0.035, y + h - 0.055, title, fontsize=13.5, fontweight="bold", color=tc, va="top")
    ax.text(x + 0.035, y + h - 0.112, body, fontsize=8.5, color=tc, va="top", linespacing=1.26)


def arrow(ax, p1, p2, color="#38bdf8"):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color,
            alpha=0.95,
        )
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("#071013")
    ax.set_facecolor("#071013")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.94,
        "C-Kernel-Engine Capability Map",
        fontsize=31,
        fontweight="bold",
        color="#f8fafc",
        va="top",
    )
    ax.text(
        0.05,
        0.885,
        "CPU-native AI runtime work: explicit kernels, generated C, measurable parity, and hardware-aware profiling.",
        fontsize=13.5,
        color="#a7f3d0",
        va="top",
    )

    rounded_box(
        ax,
        (0.05, 0.65),
        (0.28, 0.18),
        "v7: Training Foundation",
        "Backward-capable kernels\nIR train path + codegen\nPyTorch parity + finite-diff checks\nReplay determinism + safety gates\nPython authoring layer",
        "#164e63",
    )
    rounded_box(
        ax,
        (0.36, 0.65),
        (0.28, 0.18),
        "v8: Inference + Multimodal",
        "Text GGUF / safetensors bring-up\nQwen, Gemma, NanBeige paths\nQwen3-VL bridge + vision encoder\nIR visualizer + run hub\nRegression and smoke gates",
        "#14532d",
    )
    rounded_box(
        ax,
        (0.67, 0.65),
        (0.28, 0.18),
        "Kernel Surface",
        "GEMM / GEMV / Q4_K / Q8\nRMSNorm, RoPE, SwiGLU\nFlash/sliding attention\nTop-k, MoE, Mamba/DeltaNet refs\nBF16, INT8, AMX/VNNI/AVX paths",
        "#4c1d95",
    )

    rounded_box(
        ax,
        (0.05, 0.38),
        (0.28, 0.18),
        "Compiler Pipeline",
        "Family templates\nGraphIR -> LoweredIR\nKernel registry\nMemory planner / arenas\nGenerated C shared objects",
        "#7c2d12",
    )
    rounded_box(
        ax,
        (0.36, 0.38),
        (0.28, 0.18),
        "Inspectability",
        "IR reports and hub pages\nModel maps and tensor layouts\nRun artifacts in cache\nDataset / tokenizer viewers\nParity ledgers and smoke reports",
        "#0f766e",
    )
    rounded_box(
        ax,
        (0.67, 0.38),
        (0.28, 0.18),
        "Performance Loop",
        "perf + flamegraphs\nLIKWID pinned cross-vendor counters\nVTune + Advisor deep profiles\nThreadpools + memory pools\nCPU diversity: laptop, Tiny, Xeon, ARM",
        "#1e3a8a",
    )

    rounded_box(
        ax,
        (0.14, 0.12),
        (0.72, 0.13),
        "Project Thesis",
        "Do not hide the model behind a black box. Make architecture, tensors, memory, kernels, generated code, and measurements inspectable.",
        "#111827",
        ec="#22c55e",
    )

    arrow(ax, (0.33, 0.74), (0.36, 0.74))
    arrow(ax, (0.64, 0.74), (0.67, 0.74))
    arrow(ax, (0.19, 0.65), (0.19, 0.56), "#f59e0b")
    arrow(ax, (0.50, 0.65), (0.50, 0.56), "#f59e0b")
    arrow(ax, (0.81, 0.65), (0.81, 0.56), "#f59e0b")
    arrow(ax, (0.33, 0.47), (0.36, 0.47), "#34d399")
    arrow(ax, (0.64, 0.47), (0.67, 0.47), "#34d399")
    arrow(ax, (0.50, 0.38), (0.50, 0.25), "#22c55e")

    ax.text(
        0.05,
        0.035,
        "github.com/C-Kernel-Engine/C-Kernel-Engine  |  c-kernel-engine.github.io/C-Kernel-Engine  |  shivasnotes.com",
        fontsize=10.5,
        color="#94a3b8",
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
