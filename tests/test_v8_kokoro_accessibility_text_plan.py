"""Pinned English text conversion and capacity-bounded Kokoro chunk tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import prepare_kokoro_accessibility_text as planner

FIXTURE = ROOT / "tests/fixtures/tts/kokoro_accessibility_text_oracle_v1.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
VOCAB_FIXTURE = json.loads((ROOT / "tests/fixtures/tts/kokoro_accessibility_vocab_v1.json").read_text())
VOCAB = VOCAB_FIXTURE["vocab"]
assert VOCAB_FIXTURE["source_config_sha256"] == planner.PINNED_SHA256["config"]


def _tokens(case):
    return [SimpleNamespace(**token) for token in case["tokens"]]


def _case(name, status, reason="", **details):
    report = {
        "case_id": f"kokoro.accessibility-text-plan.{name}-v1",
        "name": f"Kokoro pinned text conversion {name}",
        "provider": "pinned_misaki_text_frontend",
        "dtype": "i32", "direction": "inference",
        "oracle": "pinned Kokoro KPipeline/Misaki phonemes",
        "gate": "text", "status": status,
        "blocking": status != "not_tested",
        "configuration": "af_heart; 3..40 ID range bundle; whole-context G2P",
        "reproduction_command": "python3 -m unittest "
            "tests.test_v8_kokoro_accessibility_text_plan",
    }
    if reason:
        report["reason"] = reason
    if status != "not_tested":
        report.update(max_diff=0 if status == "pass" else 1, tolerance=0)
    report.update(details)
    print("CKE_NUMERICAL_CASE " + json.dumps(report, sort_keys=True))


class KokoroAccessibilityTextPlanTest(unittest.TestCase):
    def test_committed_independent_phonemes_and_chunk_join(self):
        for case in CASES:
            with self.subTest(text=case["text"][:40]):
                tokens = _tokens(case)
                self.assertEqual(planner._phonemes(tokens),
                                 case["reference_phonemes"])
                chunks = planner.plan_tokens(case["text"], tokens, VOCAB,
                                             set(range(3, 41)))
                self.assertEqual("".join(case["text"][chunk["source_start"]:
                                         chunk["source_end"]] for chunk in chunks),
                                 case["text"])
                for chunk in chunks:
                    selected = [token for token, span in zip(tokens,
                               planner._source_spans(case["text"], tokens))
                                if span[0] >= chunk["source_start"] and
                                span[1] <= chunk["source_end"]]
                    self.assertEqual(chunk["phonemes"], planner._phonemes(selected))
                for chunk in chunks:
                    self.assertLessEqual(chunk["valid_token_count"], 40)
                    self.assertEqual(chunk["phoneme_ids"],
                                     [0] + [VOCAB[char] for char in chunk["phonemes"]] + [0])
                    self.assertEqual(chunk["voice_row"], len(chunk["phonemes"]) - 1)
        self.assertIn("words", CASES[0]["text"])
        self.assertEqual(CASES[2]["text"],
                         "A list:\n1. First item.\n2. Second item.")
        _case("committed-oracle-parity", "pass",
              fixture_sha256=planner.sha256(FIXTURE),
              cases=len(CASES))

    def test_sentence_list_and_equation_boundaries(self):
        expected = (
            ["Dr. Smith has 12 words."],
            ["The CKE runtime", "reads files and code.", "2 + 2 = 4."],
            ["A list:\n1. First item.", "2. Second item."],
        )
        for case, expected_text in zip(CASES, expected):
            with self.subTest(text=case["text"][:35]):
                chunks = planner.plan_tokens(case["text"], _tokens(case),
                                             VOCAB, set(range(3, 41)))
                self.assertEqual([chunk["source_text"] for chunk in chunks],
                                 expected_text)
        long_chunks = planner.plan_tokens(CASES[3]["text"], _tokens(CASES[3]),
                                          VOCAB, set(range(3, 41)))
        self.assertGreaterEqual(len(long_chunks), 4)
        self.assertTrue(long_chunks[-1]["source_text"].endswith("again."))

    def test_thousand_word_plan_preserves_spans_beyond_aggregate_limit(self):
        sound = next(char for char in VOCAB if char.strip()) * 6
        text = "red " * 999 + "red"
        tokens = [SimpleNamespace(text="red", whitespace=" " if index < 999 else "",
                                  phonemes=sound) for index in range(1000)]
        chunks = planner.plan_tokens(text, tokens, VOCAB, set(range(3, 41)))
        self.assertGreater(len(chunks), 64)
        self.assertLessEqual(len(chunks), planner.MAX_CHUNKS)
        self.assertEqual("".join(text[item["source_start"]:item["source_end"]]
                                 for item in chunks), text)
        self.assertEqual([item["index"] for item in chunks], list(range(len(chunks))))

    def test_pinned_long_number_and_hash_tokens_fit_opt_in_range(self):
        # Phonemes captured from the pinned full-context G2P on server_summary.txt.
        text = "35,136 ab37035fc"
        tokens = [
            SimpleNamespace(text="35,136", whitespace=" ",
                phonemes="θˈɜɹTi fˈIv θˈWzᵊnd wˈʌn hˈʌndɹəd θˈɜɹTi sˈɪks"),
            SimpleNamespace(text="ab37035fc", whitespace="",
                phonemes="ˈæb θˈɜɹTi sˈɛvən θˈWzənd θˈɜɹTi fˈIv ˌɛfsˈi"),
        ]
        with self.assertRaisesRegex(planner.TextPlanError, "35,136"):
            planner.plan_tokens(text, tokens, VOCAB, set(range(3, 41)))
        chunks = planner.plan_tokens(text, tokens, VOCAB, set(range(3, 49)))
        self.assertEqual([item["valid_token_count"] for item in chunks], [48, 46])
        self.assertEqual([item["voice_row"] for item in chunks], [45, 43])
        self.assertEqual("".join(text[item["source_start"]:item["source_end"]]
                                 for item in chunks), text)
        self.assertEqual([item["phonemes"] for item in chunks],
                         [token.phonemes for token in tokens])
        sound = next(char for char in VOCAB if char.strip())
        with self.assertRaisesRegex(planner.TextPlanError, "no supported chunk"):
            planner.plan_tokens("long", [SimpleNamespace(text="long", whitespace="",
                                                       phonemes=sound * 47)],
                                VOCAB, set(range(3, 49)))

    def test_opt_in_range_reserves_long_requests_for_single_tokens(self):
        sound = next(char for char in VOCAB if char.strip()) * 6
        text = "red " * 9 + "red"
        tokens = [SimpleNamespace(text="red", whitespace=" " if index < 9 else "",
                                  phonemes=sound) for index in range(10)]
        chunks = planner.plan_tokens(text, tokens, VOCAB, set(range(3, 49)))
        self.assertTrue(all(chunk["valid_token_count"] <=
                            planner.PREFERRED_REQUEST_IDS for chunk in chunks))
        self.assertEqual("".join(text[chunk["source_start"]:chunk["source_end"]]
                                 for chunk in chunks), text)

    def test_rejects_dropped_source_and_unsupported_geometry(self):
        with self.assertRaisesRegex(planner.TextPlanError, "unspoken source token"):
            planner.plan_tokens("🧪", [SimpleNamespace(text="🧪", whitespace="",
                                                      phonemes="")], VOCAB,
                                set(range(3, 41)))
        case = CASES[0]
        with self.assertRaisesRegex(planner.TextPlanError, "source text"):
            planner.plan_tokens(case["text"] + "!", _tokens(case), VOCAB,
                                set(range(3, 41)))
        with self.assertRaisesRegex(planner.TextPlanError, "no supported chunk"):
            planner.plan_tokens(case["text"], _tokens(case), VOCAB, {8})
        with self.assertRaisesRegex(planner.TextPlanError, "exceeds the native"):
            planner.plan_tokens(case["text"], _tokens(case), VOCAB, {49})
        with self.assertRaisesRegex(planner.TextPlanError, "UTF-8 bytes"):
            planner.plan_tokens("x" * (planner.MAX_TEXT_BYTES + 1), _tokens(case),
                                VOCAB, set(range(3, 41)))
        with self.assertRaisesRegex(planner.TextPlanError, "unsupported phonemes"):
            planner.plan_tokens("x", [SimpleNamespace(text="x", whitespace="",
                                                      phonemes="♠")], VOCAB,
                                set(range(3, 41)))

    def test_asset_coverage_is_explicit(self):
        requested = bool(os.environ.get("CKE_TTS_RUN_PINNED_ACCESSIBILITY_TEXT"))
        required = ("CKE_TTS_ACCESSIBILITY_CONFIG", "CKE_TTS_ACCESSIBILITY_BUNDLE")
        missing = [name for name in required if not os.environ.get(name)]
        if requested and not missing:
            return
        reason = ("pinned text assets not requested" if not requested else
                  "missing assets: " + ", ".join(missing))
        _case("live-pinned-oracle", "not_tested", reason)

    def test_generated_audio_coverage_is_explicit(self):
        requested = bool(os.environ.get("CKE_TTS_RUN_ACCESSIBILITY_TEXT_AUDIO"))
        required = ("CKE_TTS_ACCESSIBILITY_CONFIG", "CKE_TTS_ACCESSIBILITY_BUNDLE")
        missing = [name for name in required if not os.environ.get(name)]
        if requested and not missing:
            return
        reason = ("generated audio asset run not requested" if not requested else
                  "missing assets: " + ", ".join(missing))
        _case("generated-two-chunk-audio", "not_tested",
              reason,
              provider="native_kokoro_dispatch", gate="execution",
              oracle="checked PCM16 geometry; waveform parity separate")

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_PINNED_ACCESSIBILITY_TEXT"),
                         "pinned Misaki/Kokoro text environment not requested")
    def test_live_pinned_oracle_and_bundle(self):
        required = ("CKE_TTS_ACCESSIBILITY_CONFIG", "CKE_TTS_ACCESSIBILITY_BUNDLE")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            self.skipTest("missing assets: " + ", ".join(missing))
        for case in CASES:
            with self.subTest(text=case["text"][:40]):
                plan = planner.prepare(
                    case["text"], Path(os.environ[required[0]]),
                    Path(os.environ[required[1]]))
                self.assertEqual(plan["full_reference_phonemes"],
                                 case["reference_phonemes"])
                self.assertEqual(plan["source_text"], case["text"])
        with self.assertRaisesRegex(planner.TextPlanError, "unspoken source token"):
            planner.prepare("The price is $1.25.", Path(os.environ[required[0]]),
                            Path(os.environ[required[1]]))
        _case("live-pinned-oracle", "pass", cases=len(CASES))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_ACCESSIBILITY_TEXT_AUDIO"),
                         "generated two-chunk audio not requested")
    def test_generated_two_chunk_audio(self):
        required = ("CKE_TTS_ACCESSIBILITY_CONFIG", "CKE_TTS_ACCESSIBILITY_BUNDLE")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            self.skipTest("missing assets: " + ", ".join(missing))
        bundle = Path(os.environ[required[1]])
        dispatcher = bundle / "native-kokoro-dispatch"
        text = "Dr. Smith has 12 words. Then check 2 + 2 = 4."
        plan = planner.prepare(text, Path(os.environ[required[0]]), bundle)
        self.assertEqual([chunk["source_text"] for chunk in plan["chunks"]],
                         ["Dr. Smith has 12 words.", "Then check 2 + 2 = 4."])
        sample_counts = []
        with tempfile.TemporaryDirectory(prefix="cke-tts-text-audio-") as temporary:
            for chunk in plan["chunks"]:
                output = Path(temporary) / f"chunk-{chunk['index']}.wav"
                command = [str(dispatcher), str(bundle), str(output), "7",
                           ",".join(map(str, chunk["phoneme_ids"]))]
                completed = subprocess.run(command, capture_output=True,
                                           text=True, timeout=180, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                with wave.open(str(output), "rb") as stream:
                    self.assertEqual((stream.getnchannels(), stream.getsampwidth(),
                                      stream.getframerate()), (1, 2, 24000))
                    samples = stream.getnframes()
                    self.assertGreater(samples, 0)
                    self.assertEqual(len(stream.readframes(samples)), samples * 2)
                    sample_counts.append(samples)
        _case("generated-two-chunk-audio", "pass",
              provider="native_kokoro_dispatch", gate="execution",
              oracle="checked PCM16 geometry; waveform parity separate",
              sample_counts=sample_counts,
              bundle_manifest_sha256=plan["bundle_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
