import os
import platform
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_q8_k_quantizer_preserves_output_red_zones() -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("Q8_K SSE provider is an x86 kernel")

    compiler = os.environ.get("CC", "cc")
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "q8_k_write_bounds"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-msse4.1",
                "-I",
                str(ROOT / "include"),
                str(ROOT / "tests" / "test_v8_quantized_write_bounds.c"),
                str(ROOT / "src" / "kernels" / "quantize_row_q8_k_sse.c"),
                "-lm",
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run([str(executable)], check=True, capture_output=True, text=True)
