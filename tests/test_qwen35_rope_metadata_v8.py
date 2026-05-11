import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from convert_gguf_to_bump_v8 import extract_mrope_sections_for_arch  # type: ignore


class Qwen35RopeMetadataV8Tests(unittest.TestCase):
    def test_qwen35_reads_own_rope_dimension_sections(self) -> None:
        meta = {
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "qwen3vl.rope.dimension_sections": [1, 1, 1, 1],
        }

        self.assertEqual(extract_mrope_sections_for_arch(meta, "qwen35"), [11, 11, 10, 0])

    def test_qwen3vl_keeps_qwen3vl_rope_dimension_sections(self) -> None:
        meta = {
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "qwen3vl.rope.dimension_sections": [16, 24, 24, 0],
        }

        self.assertEqual(extract_mrope_sections_for_arch(meta, "qwen3vl"), [16, 24, 24, 0])

    def test_invalid_sections_are_ignored(self) -> None:
        self.assertIsNone(extract_mrope_sections_for_arch({"qwen35.rope.dimension_sections": [1, 2, 3]}, "qwen35"))


if __name__ == "__main__":
    unittest.main()
