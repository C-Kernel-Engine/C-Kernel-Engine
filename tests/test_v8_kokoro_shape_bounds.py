"""Compile and exercise Kokoro's checked dynamic-output planner."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "kernels"
INCLUDE = ROOT / "include"


class KokoroShapeBoundsTest(unittest.TestCase):
    def test_native_shape_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "test_kokoro_shape_bounds"
            subprocess.run(
                [
                    "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I", str(INCLUDE),
                    str(SOURCE / "kokoro_shape_bounds.c"),
                    str(SOURCE / "audio_duration_expand.c"),
                    str(ROOT / "tests" / "test_v8_kokoro_shape_bounds.c"),
                    "-o", str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
