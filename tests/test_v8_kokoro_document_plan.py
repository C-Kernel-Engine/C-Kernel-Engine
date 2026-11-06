"""Asset-free source-coverage contract for long Kokoro documents."""

import hashlib
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
from prepare_kokoro_document import (  # noqa: E402
    MAX_DOCUMENT_BYTES, SCHEMA, TextPlanError, prepare_document, text_windows)


def fake_bounded_frontend(fragment, _config, _bundle, *, technical_mode):
    source = fragment.strip()
    if "UNSUPPORTED" in source:
        raise TextPlanError("unsupported technical token")
    return {"source_text": source, "language": "en-us", "voice": "af_heart",
            "bundle_manifest_sha256": "manifest", "bundle_identity_sha256": "bundle",
            "frontend_sha256": {"mock": True},
            "pronunciation_expansions": [],
            "chunks": [{"index": 0, "source_start": 0,
                        "source_end": len(source), "source_text": source,
                        "phoneme_ids": [0, 1, 0], "valid_token_count": 3,
                        "voice_row": 0, "technical_mode": technical_mode}]}


class KokoroDocumentPlanTests(unittest.TestCase):
    def plan(self, text):
        return prepare_document(text, Path("config"), Path("bundle"),
                                prepare_window=fake_bounded_frontend)

    def test_ten_thousand_words_preserve_every_source_span(self):
        sentence = "CKE reads PR 862 and issue 833 from /data/report.txt. "
        text = " \n" + sentence * 1250 + "\n"
        self.assertGreaterEqual(len(text.split()), 10_000)
        result = self.plan(text)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["source_sha256"],
                         hashlib.sha256(text.encode()).hexdigest())
        self.assertGreater(len(result["windows"]), 1)
        self.assertEqual("".join(text[row["source_start"]:row["source_end"]]
                                 for row in result["chunks"]), text)
        self.assertEqual([row["index"] for row in result["chunks"]],
                         list(range(len(result["chunks"]))))
        self.assertTrue(all(len(text[row["source_start"]:row["source_end"]]
                                .encode("utf-8")) <= 8192
                            for row in result["windows"]))
        self.assertEqual(result["bundle_identity_sha256"], "bundle")

    def test_utf8_sentence_and_whitespace_boundaries(self):
        text = "  Café one. Café two. Café three.  "
        windows = text_windows(text, max_bytes=15)
        self.assertEqual("".join(text[a:b] for a, b in windows), text)
        self.assertTrue(all(len(text[a:b].encode()) <= 15 for a, b in windows))
        result = self.plan(text)
        self.assertEqual(result["chunks"][0]["source_start"], 0)
        self.assertEqual(result["chunks"][-1]["source_end"], len(text))

    def test_oversized_word_and_unsupported_span_fail_explicitly(self):
        with self.assertRaisesRegex(TextPlanError, "no safe document boundary"):
            text_windows("x" * 200, max_bytes=32)
        with self.assertRaisesRegex(TextPlanError, "document source span.*unsupported"):
            self.plan("A sentence. " * 800 + "UNSUPPORTED token.")
        with self.assertRaisesRegex(TextPlanError, "document exceeds"):
            text_windows("x " * (MAX_DOCUMENT_BYTES // 2 + 1))

    def test_identity_changes_are_rejected(self):
        text = "A sentence. " * 900
        calls = 0

        def changing(fragment, config, bundle, *, technical_mode):
            nonlocal calls
            calls += 1
            result = fake_bounded_frontend(fragment, config, bundle,
                                           technical_mode=technical_mode)
            result["bundle_identity_sha256"] = f"bundle-{calls}"
            return result

        with self.assertRaisesRegex(TextPlanError, "identity changed"):
            prepare_document(text, Path("config"), Path("bundle"),
                             prepare_window=changing)


if __name__ == "__main__":
    unittest.main()
