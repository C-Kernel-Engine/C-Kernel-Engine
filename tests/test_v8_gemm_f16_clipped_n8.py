from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckernel_engine.h"

static uint32_t bits(float x) {
    uint32_t result;
    memcpy(&result, &x, sizeof(result));
    return result;
}

static int run_case(int M, int N, int K, int use_bias, int use_clamp) {
    float *A = malloc((size_t)M * K * sizeof(float));
    uint16_t *W = malloc((size_t)N * K * sizeof(uint16_t));
    float *bias = malloc((size_t)N * sizeof(float));
    float *baseline = malloc(((size_t)M * N + 2) * sizeof(float));
    float *candidate = malloc(((size_t)M * N + 2) * sizeof(float));
    if (!A || !W || !bias || !baseline || !candidate) return 2;
    const uint16_t half_bits[] = {0x3c00, 0xbc00, 0x3555, 0x3800, 0x0001, 0x7bff, 0x0400};
    for (int i = 0; i < M * K; ++i) {
        A[i] = ((float)((i * 37) % 89) - 44.0f) / 19.0f;
    }
    if (M * K > 4) {
        A[0] = 1.00048828125f;
        A[1] = -1.00048828125f;
        A[2] = 1.5f;
        A[3] = -1.5f;
    }
    for (int i = 0; i < N * K; ++i) W[i] = half_bits[(i * 13) % 7];
    for (int n = 0; n < N; ++n) bias[n] = (float)(n % 11 - 5) * 0.03125f;
    for (int i = 0; i < M * N + 2; ++i) {
        baseline[i] = -12345.0f;
        candidate[i] = -12345.0f;
    }
    const float in_min = -1.0f, in_max = 1.0f;
    const float out_min = -3.0f, out_max = 3.0f;
    const float *b = use_bias ? bias : NULL;
    const float *imin = use_clamp ? &in_min : NULL;
    const float *imax = use_clamp ? &in_max : NULL;
    const float *omin = use_clamp ? &out_min : NULL;
    const float *omax = use_clamp ? &out_max : NULL;
    gemm_nt_f16_clipped(A, W, b, imin, imax, omin, omax, baseline + 1, M, N, K);
    gemm_nt_f16_clipped_n8(A, W, b, imin, imax, omin, omax, candidate + 1, M, N, K);
    for (int i = 0; i < M * N + 2; ++i) {
        if (bits(baseline[i]) != bits(candidate[i])) {
            fprintf(stderr, "mismatch M=%d N=%d K=%d index=%d baseline=%08x candidate=%08x\n",
                    M, N, K, i, bits(baseline[i]), bits(candidate[i]));
            return 1;
        }
    }
    free(A); free(W); free(bias); free(baseline); free(candidate);
    return 0;
}

int main(void) {
    const int shapes[][3] = {
        {1, 1, 1}, {2, 7, 9}, {3, 8, 16}, {4, 9, 17},
        {13, 31, 63}, {17, 65, 128}, {8, 96, 256}
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        for (int bias = 0; bias < 2; ++bias) {
            for (int clamp = 0; clamp < 2; ++clamp) {
                const int rc = run_case(shapes[s][0], shapes[s][1], shapes[s][2], bias, clamp);
                if (rc) return rc;
            }
        }
    }
    return 0;
}
"""


class ClippedF16N8Tests(unittest.TestCase):
    def test_compiled_candidate_matches_baseline_bits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_f16_clipped_n8_") as tmp:
            source = Path(tmp) / "fixture.c"
            source.write_text(SOURCE, encoding="ascii")
            builds = [
                ("scalar", [], (1,)),
                ("openmp", ["-fopenmp"], (1, 2, 8)),
            ]
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.is_file():
                flags = cpuinfo.read_text(encoding="ascii", errors="ignore")
                if all(flag in flags for flag in ("avx2", "f16c", "fma")):
                    builds.append(("avx2", ["-fopenmp", "-mavx2", "-mf16c", "-mfma"], (1, 2, 8)))
            for name, flags, thread_counts in builds:
                executable = Path(tmp) / name
                subprocess.run(
                    ["cc", "-std=c11", "-O3", *flags, "-ffunction-sections",
                     "-fdata-sections", "-I", str(ROOT / "include"),
                     str(ROOT / "src/kernels/gemm_kernels_f16.c"), str(source),
                     "-Wl,--gc-sections", "-ldl", "-lm", "-o", str(executable)],
                    check=True, capture_output=True, text=True,
                )
                for threads in thread_counts:
                    with self.subTest(build=name, threads=threads):
                        subprocess.run(
                            [str(executable)], check=True, capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
                        )

    def test_map_owned_generated_call_contract(self) -> None:
        import json

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import load_kernel_call_abis

        circuit = json.loads((ROOT / "version/v8/circuits/gemma4_vision.json").read_text())
        selected = circuit["kernels"]
        for op in ("q_proj", "k_proj", "v_proj", "out_proj", "mlp_gate_up", "mlp_down", "projector_fc2"):
            with self.subTest(op=op):
                self.assertEqual(selected[op], "gemm_nt_f16_clipped_n8")
        binding = load_kernel_call_abis()["gemm_nt_f16_clipped_n8"]
        params = binding["call_abi"]["params"]
        self.assertEqual(binding["function"], "gemm_nt_f16_clipped_n8")
        self.assertEqual(
            [param["name"] for param in params],
            ["A", "B", "bias", "input_min", "input_max", "output_min", "output_max", "C", "M", "N", "K"],
        )
        self.assertEqual([param["source"] for param in params[3:7]], [
            "weight_f:_input_min", "weight_f:_input_max",
            "weight_f:_output_min", "weight_f:_output_max",
        ])


if __name__ == "__main__":
    unittest.main()
