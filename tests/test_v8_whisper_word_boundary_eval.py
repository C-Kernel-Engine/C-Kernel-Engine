"""Transcription and boundary metrics stay independent for word timing."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "version/v8/scripts" /
          "evaluate_whisper_word_boundaries_v8.py")
SPEC = importlib.util.spec_from_file_location("whisper_boundary_eval", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)
ASSEMBLY_SPEC = importlib.util.spec_from_file_location(
    "whisper_word_assembly", SCRIPT.with_name("assemble_whisper_word_timings_v8.py"))
assembly = importlib.util.module_from_spec(ASSEMBLY_SPEC)
ASSEMBLY_SPEC.loader.exec_module(assembly)


def reference(words):
    return {"id": "synthetic", "words": [
        {"word": word, "start_ms": start * 1000, "end_ms": end * 1000}
        for word, start, end in words]}


def alignment(words):
    return {"words": [
        {"text": word, "start_seconds": start, "end_seconds": end}
        for word, start, end in words]}


class WhisperBoundaryEvalTest(unittest.TestCase):
    def test_exact_repeated_words_retain_order_and_boundary_error(self):
        expected = reference([("go", 0, .2), ("go", .3, .5), ("now", .6, .9)])
        actual = alignment([("Go", .02, .22), ("go", .32, .52), ("now", .62, .92)])
        result = evaluator.evaluate(expected, actual)
        self.assertEqual(result["transcription"]["match"], 3)
        self.assertEqual(result["transcription"]["word_error_rate"], 0)
        self.assertEqual(result["boundaries"]["matched_words_with_timing"], 3)
        self.assertAlmostEqual(result["boundaries"]["combined_error_ms"]["median"], 20)
        self.assertEqual(result["boundaries"]["worst"]["word"], "go")

    def test_omitted_fillers_are_transcription_failures_not_timing_matches(self):
        expected = reference([("um", 0, .2), ("uh", .4, .5),
                              ("start", .6, .9), ("now", 1.0, 1.2)])
        actual = alignment([("start", 0, .9), ("now", 1.0, 1.2)])
        result = evaluator.evaluate(expected, actual)
        self.assertEqual(result["transcription"]["deletion"], 2)
        self.assertEqual(result["fillers"]["reference_count"], 2)
        self.assertEqual(result["fillers"]["matched_count"], 0)
        self.assertEqual([entry["word"] for entry in result["fillers"]["omitted"]],
                         ["um", "uh"])
        self.assertEqual(result["boundaries"]["matched_words_with_timing"], 2)
        self.assertEqual(result["boundaries"]["start_error_ms"]["maximum"], 600)

    def test_missing_repeated_word_excludes_ambiguous_timing_match(self):
        expected = reference([("go", 0, .2), ("go", .3, .5), ("now", .6, .9)])
        actual = alignment([("go", 0, .5), ("now", .6, .9)])
        result = evaluator.evaluate(expected, actual)
        self.assertEqual(result["transcription"]["deletion"], 1)
        self.assertEqual(result["boundaries"]["matched_words_with_timing"], 1)
        self.assertEqual(result["boundaries"]["matched_words_excluded_ambiguous_repeat"], [0])

    def test_substitution_insertion_and_missing_boundaries_are_distinct(self):
        expected = reference([("read", 0, .2), ("three", .3, .6)])
        actual = alignment([("read", None, None), ("tree", .3, .6),
                            ("please", .7, .9)])
        result = evaluator.evaluate(expected, actual)
        self.assertEqual(result["transcription"]["substitution"], 1)
        self.assertEqual(result["transcription"]["insertion"], 1)
        self.assertEqual(result["transcription"]["word_error_rate"], 1.0)
        self.assertEqual(result["boundaries"]["matched_words_missing_timing"], [0])
        self.assertIsNone(result["boundaries"]["combined_error_ms"]["median"])

    def test_invalid_boundary_is_rejected(self):
        expected = reference([("word", 0, .2)])
        with self.assertRaisesRegex(ValueError, "invalid predicted"):
            evaluator.evaluate(expected, alignment([("word", .4, .1)]))
        with self.assertRaisesRegex(ValueError, "empty or nonlexical"):
            evaluator.evaluate(expected, alignment([("!", .0, .1)]))

    def test_trace_must_belong_to_the_same_audio_window(self):
        report = {"wav_sha256": "audio", "segments": [
            {"decoder": {"attention_trace": {"sha256": "first"}}},
            {"decoder": {"attention_trace": {"sha256": "second"}}},
        ]}
        aligned = {"schema": "cke.whisper_word_alignment.v1",
                   "window": {"index": 1},
                   "provenance": {"trace_sha256": "second"}}
        self.assertEqual(evaluator.validate_provenance(report, aligned, "audio"), "second")
        aligned["provenance"]["trace_sha256"] = "first"
        with self.assertRaisesRegex(ValueError, "audio window"):
            evaluator.validate_provenance(report, aligned, "audio")
        aligned["provenance"]["trace_sha256"] = "second"
        with self.assertRaisesRegex(ValueError, "audio hash"):
            evaluator.validate_provenance(report, aligned, "different")


class WhisperWindowAssemblyTest(unittest.TestCase):
    def setUp(self):
        self.report_hash = "c" * 64
        self.audio_hash = "d" * 64
        self.report = {"wav_sha256": self.audio_hash, "segments": [
            {"index": index, "source_frame_start": start,
             "source_frame_consumed_end": consumed,
             "source_frame_window_end": window_end,
             "encoder": {"audio": {"source_sample_rate": 100,
                                    "source_frames": 1000,
                                    "window_source_frames": 600,
                                    "window_start_frame": start}},
             "decoder": {"attention_trace": {"sha256": str(index + 1) * 64}}}
            for index, start, consumed, window_end in
            ((0, 0, 400, 600), (1, 400, 1000, 1000))]}

    def aligned(self, index, words):
        return {"schema": "cke.whisper_word_alignment.v1", "status": "aligned",
                "window": {"index": index, "start_seconds": index * 4.0,
                           "source_frames": 600, "source_sample_rate": 100},
                "provenance": {"trace_sha256": str(index + 1) * 64,
                               "run_report_sha256": self.report_hash,
                               "audio_sha256": self.audio_hash},
                "words": [{"text": text, "start_seconds": start,
                           "end_seconds": end, "audio_window": index}
                          for text, start, end in words]}

    def test_commit_intervals_preserve_real_repetition_and_expose_overlap(self):
        first = self.aligned(0, [("go", 1.0, 1.2), ("go", 2.0, 2.2),
                                 ("seam", 3.9, 4.1), ("echo", 4.3, 4.5)])
        second = self.aligned(1, [("seam", 4.1, 4.3), ("echo", 4.3, 4.5),
                                  ("go", 7.0, 7.2)])
        result = assembly.assemble(self.report, [first, second], self.report_hash)
        self.assertEqual([word["text"] for word in result["words"]],
                         ["go", "go", "seam", "echo", "go"])
        self.assertEqual([word["assembly_review_reasons"] for word in
                          result["review_words"]],
                         [["crosses_consumed_boundary"],
                          ["unconsumed_window_overlap"]])
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["edit_ready"])

    def test_long_silence_span_and_missing_word_timing_require_review(self):
        self.report["segments"] = [{
            "index": 0, "source_frame_start": 0,
            "source_frame_consumed_end": 1000, "source_frame_window_end": 1000,
            "encoder": {"audio": {"source_sample_rate": 100,
                                   "source_frames": 1000,
                                   "window_source_frames": 1000,
                                   "window_start_frame": 0}},
            "decoder": {"attention_trace": {"sha256": "1" * 64}}}]
        aligned = self.aligned(0, [("Actually", 0.0, 5.44),
                                   ("I", None, None), ("think", 5.6, 5.72)])
        aligned["window"]["source_frames"] = 1000
        result = assembly.assemble(self.report, [aligned], self.report_hash)
        self.assertEqual([word["text"] for word in result["words"]], ["think"])
        self.assertEqual([word["assembly_review_reasons"] for word in
                          result["review_words"]],
                         [["long_alignment_span"], ["missing_word_timing"]])

    def test_wrong_provenance_geometry_and_missing_window_fail_closed(self):
        first = self.aligned(0, [("one", 1.0, 1.2)])
        second = self.aligned(1, [("two", 7.0, 7.2)])
        with self.assertRaisesRegex(ValueError, "every audio window"):
            assembly.assemble(self.report, [first], self.report_hash)
        second["provenance"]["trace_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            assembly.assemble(self.report, [first, second], self.report_hash)
        second["provenance"]["trace_sha256"] = "2" * 64
        second["window"]["start_seconds"] = 0.0
        with self.assertRaisesRegex(ValueError, "geometry"):
            assembly.assemble(self.report, [first, second], self.report_hash)
        second["window"]["start_seconds"] = 4.0
        first["words"].append({"text": "earlier", "start_seconds": .5,
                               "end_seconds": .7, "audio_window": 0})
        with self.assertRaisesRegex(ValueError, "backward"):
            assembly.assemble(self.report, [first, second], self.report_hash)


if __name__ == "__main__":
    unittest.main()
