"""Compile and execute Q4_K/Q8_K two-row decode arithmetic by ISA lane."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _cpu_flags() -> set[str]:
    if platform.system() != "Linux":
        return set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.partition(":")[2].split())
    except OSError:
        pass
    return set()


class Q4KDecodeM2Tests(unittest.TestCase):
    def test_packed_two_row_decode_matches_one_row_contract(self) -> None:
        flags = _cpu_flags()
        lanes = [
            ("portable", (), True),
        ]
        if platform.machine().lower() in {"x86_64", "amd64", "i386", "i686"}:
            lanes.extend([
                ("avx2", ("-mavx2", "-mfma", "-mf16c"),
                 {"avx2", "fma", "f16c"} <= flags),
                ("avx512", ("-mavx512f", "-mavx512bw", "-mavx512vl",
                            "-mavx512vnni", "-mavx2", "-mfma", "-mf16c"),
                 {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "fma", "f16c"} <= flags),
            ])
        with tempfile.TemporaryDirectory(prefix="cke-q4k-m2-") as directory:
            for name, isa, supported in lanes:
                with self.subTest(isa=name):
                    binary = Path(directory) / name
                    command = [
                        os.environ.get("CC", "cc"), "-O2", "-std=c11",
                        "-ffunction-sections", "-fdata-sections", *isa,
                        "-I", str(ROOT / "include"),
                        "-Wl,--gc-sections",
                        str(ROOT / "src/kernels/gemm_kernels_q4k_q8k_vnni.c"),
                        str(ROOT / "tests/test_q4k_q8k_decode_m2.c"),
                        "-lm", "-o", str(binary),
                    ]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    if not supported:
                        self.skipTest(f"{name} ISA unavailable on this runner")
                    result = subprocess.run([str(binary)], check=True,
                                            capture_output=True, text=True)
                    self.assertIn("PASS", result.stdout)
                    print(f"{name} Q4_K/Q8_K M=2 lane: EXECUTED PASS")


if __name__ == "__main__":
    unittest.main()
