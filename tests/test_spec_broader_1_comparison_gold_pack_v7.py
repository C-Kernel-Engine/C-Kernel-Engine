import json
import tempfile
import unittest
from pathlib import Path

from version.v7.scripts.bootstrap_spec_broader_1_comparison_gold_pack_v7 import CASES, write_gold_pack
from version.v7.scripts.render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg


class SpecBroader1ComparisonGoldPackTests(unittest.TestCase):
    def test_cases_compile_with_bound_content(self) -> None:
        for case in CASES:
            svg = render_structured_scene_spec09_svg(case.scene_text, content=case.content_json)
            self.assertTrue(svg.startswith("<svg"))
            self.assertTrue(svg.endswith("</svg>"))
            headline = case.content_json["title"]["headline"]
            for token in headline.split()[:2]:
                self.assertIn(token, svg)
            self.assertIn(case.content_json["bars"]["primary"]["label"], svg)
            self.assertIn(case.content_json["bars"]["secondary"]["label"], svg)

    def test_write_gold_pack_emits_scene_and_content_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            status = write_gold_pack(out_dir)
            self.assertEqual(status["family"], "comparison_span_chart")
            self.assertEqual(len(status["cases"]), 3)
            for row in status["cases"]:
                scene_path = Path(tmp) / Path(row["scene_dsl"]).name
                content_path = Path(tmp) / Path(row["content_json"]).name
                self.assertTrue(scene_path.exists())
                self.assertTrue(content_path.exists())
                content = json.loads(content_path.read_text(encoding="utf-8"))
                self.assertIn("title", content)


if __name__ == "__main__":
    unittest.main()
