from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"


class VisionStatusPageTests(unittest.TestCase):
    def test_published_page_matches_source_and_keeps_scope_visible(self) -> None:
        source = (SITE / "_pages" / "v8-vision-status.html").read_text(encoding="utf-8")
        published = (SITE / "v8-vision-status.html").read_text(encoding="utf-8")
        body = "\n".join(
            line for line in source.splitlines()
            if not re.fullmatch(r"<!--.*-->", line)
        ).strip()

        self.assertIn(body, published)
        for artifact in ("Qwen3-VL", "Qwen3.6-VL", "Gemma4", "Cohere Compass"):
            self.assertIn(artifact, body)
        for boundary in ("Encoder vs llama.cpp", "Decoder vs llama.cpp", "PyTorch",
                         "Image tasks", "HTTP / harness", "First unresolved boundary"):
            self.assertIn(boundary, body)
        self.assertIn("No full vision artifact is certified", body)
        self.assertNotIn("/data/cke/", body)
        self.assertNotIn("/home/", body)

    def test_page_is_discoverable(self) -> None:
        for path in (SITE / "_pages" / "architecture-links.html",
                     SITE / "architecture-links.html", SITE / "sitemap.xml"):
            self.assertIn("v8-vision-status.html", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
