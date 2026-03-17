#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

from render_svg_structured_atoms_v7 import render_structured_svg_atoms


class StructuredSvgAtomsRendererTest(unittest.TestCase):
    def test_renders_single_shape(self) -> None:
        svg = render_structured_svg_atoms(
            "[svg] [w:128] [h:128] [bg:paper] [layout:single] "
            "[circle] [cx:64] [cy:64] [r:18] [fill:blue] [stroke:black] [sw:2] [/svg]"
        )
        self.assertIn("<svg", svg)
        self.assertIn("<circle", svg)
        self.assertIn('fill="#3b82f6"', svg)

    def test_renders_pair_layout(self) -> None:
        svg = render_structured_svg_atoms(
            "[svg] [w:128] [h:128] [bg:mint] [layout:pair-h] "
            "[circle] [cx:36] [cy:64] [r:16] [fill:red] [stroke:black] [sw:2] "
            "[rect] [x:74] [y:52] [width:32] [height:24] [rx:6] [fill:green] [stroke:black] [sw:2] "
            "[/svg]"
        )
        self.assertIn("<circle", svg)
        self.assertIn("<rect", svg)
        self.assertEqual(svg.count("<rect"), 2)  # background + foreground rect

    def test_renders_text_card(self) -> None:
        svg = render_structured_svg_atoms(
            "[svg] [w:128] [h:128] [bg:slate] [layout:label-card] "
            "[rect] [x:18] [y:38] [width:92] [height:44] [rx:8] [fill:orange] [stroke:white] [sw:2] "
            "[text] [tx:64] [ty:64] [font:14] [anchor:middle] [fill:white] DATA [/text] "
            "[/svg]"
        )
        self.assertIn("<text", svg)
        self.assertIn(">DATA</text>", svg)
        self.assertIn('fill="#1f2937"', svg)


if __name__ == "__main__":
    unittest.main()
