"""Non-destructive Whisper filler proposal and explicit approval contracts."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave


SCRIPT = (Path(__file__).resolve().parents[1] / "version/v8/scripts" /
          "review_whisper_fillers_v8.py")
SPEC = importlib.util.spec_from_file_location("whisper_filler_review", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "whisper_boundary_eval", SCRIPT.with_name("evaluate_whisper_word_boundaries_v8.py"))
evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


class WhisperFillerReviewTest(unittest.TestCase):
    def test_filler_vocabulary_matches_boundary_evaluator(self):
        self.assertEqual(review.FILLERS, evaluator.FILLERS)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "source.wav"
        samples = [index % 1000 - 500 for index in range(2000)]
        self.samples = samples
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(1000)
            output.writeframes(struct.pack(f"{len(samples)}h", *samples))
        self.source_hash = review.sha256(self.audio)
        self.run_report = self.root / "run-report.json"
        self.run_report.write_text(json.dumps({"wav_sha256": self.source_hash}))
        self.assembly = self.root / "assembly.json"
        self.assembly.write_text(json.dumps({
            "schema": "cke.whisper_word_assembly.v1", "edit_ready": False,
            "provenance": {"audio_sha256": self.source_hash,
                           "run_report_sha256": review.sha256(self.run_report)},
            "words": [{"text": "Um", "start_seconds": .2,
                       "end_seconds": .35, "audio_window": 0,
                       "window_word_index": 0}],
            "review_words": [{"text": "uh", "start_seconds": None,
                              "end_seconds": None, "audio_window": 0,
                              "window_word_index": 1,
                              "assembly_review_reasons": ["missing_word_timing"]}],
        }))
        self.manual = self.root / "manual.json"
        self.manual.write_text(json.dumps({
            "schema": "cke.whisper_manual_filler_flags.v1",
            "audio_sha256": self.source_hash,
            "flags": [{"label": "uh", "start_seconds": .9,
                       "end_seconds": 1.05}],
        }))

    def proposals(self):
        directory = self.root / "review"
        result = review.propose(self.audio, self.run_report, self.assembly,
                                directory, self.manual)
        return directory, result

    def approval(self, directory, result, indexes=(0,)):
        proposal_path = directory / "proposals.json"
        approvals = [{"id": result["candidates"][index]["id"],
                      "reviewed_preview_sha256": result["candidates"][index]["preview_sha256"],
                      "cut_start_seconds": .22 if index == 0 else .92,
                      "cut_end_seconds": .32 if index == 0 else 1.02}
                     for index in indexes]
        path = self.root / "approval.json"
        path.write_text(json.dumps({
            "schema": "cke.whisper_filler_approval.v1",
            "audio_sha256": self.source_hash,
            "proposal_sha256": review.sha256(proposal_path),
            "approved_by": "test reviewer", "approved": approvals,
        }))
        return path

    def test_propose_and_approved_render_preserve_source(self):
        directory, result = self.proposals()
        self.assertEqual(result["status"], "AWAITING_HUMAN_REVIEW")
        self.assertIsNone(result["edited_audio"])
        self.assertEqual([item["source"] for item in result["candidates"]],
                         ["transcribed_filler", "manual_flag"])
        self.assertEqual(len(result["unlocated_transcribed_fillers"]), 1)
        for candidate in result["candidates"]:
            preview = directory / candidate["preview_file"]
            self.assertEqual(review.sha256(preview), candidate["preview_sha256"])
            with wave.open(str(preview), "rb") as opened:
                self.assertEqual(opened.getnframes(),
                                 candidate["context_frames"][1] -
                                 candidate["context_frames"][0])
        approval = self.approval(directory, result, indexes=(0, 1))
        output = self.root / "edited.wav"
        evidence = self.root / "edited.json"
        rendered = review.render(self.audio, directory / "proposals.json",
                                 approval, output, evidence)
        self.assertEqual(rendered["output_frames"], 1800)
        self.assertEqual(rendered["source_audio_sha256"], self.source_hash)
        self.assertEqual(review.sha256(self.audio), self.source_hash)
        with wave.open(str(output), "rb") as opened:
            self.assertEqual(opened.getnframes(), 1800)
            self.assertEqual(opened.getframerate(), 1000)
            observed = struct.unpack("1800h", opened.readframes(1800))
        self.assertEqual(observed, tuple(self.samples[:220] + self.samples[320:920] +
                                         self.samples[1020:]))
        self.assertEqual(json.loads(evidence.read_text())["output_audio_sha256"],
                         review.sha256(output))

    def test_wrong_identity_missing_approval_and_overlap_reject_before_output(self):
        directory, result = self.proposals()
        approval = self.approval(directory, result)
        output = self.root / "edited.wav"
        evidence = self.root / "edited.json"
        data = json.loads(approval.read_text())
        data["proposal_sha256"] = "0" * 64
        approval.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "approval"):
            review.render(self.audio, directory / "proposals.json", approval,
                          output, evidence)
        self.assertFalse(output.exists())
        data["proposal_sha256"] = review.sha256(directory / "proposals.json")
        data["approved_by"] = ""
        approval.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "approval"):
            review.render(self.audio, directory / "proposals.json", approval,
                          output, evidence)
        data["approved_by"] = "test reviewer"
        data["approved"] = [dict(data["approved"][0]), dict(data["approved"][0])]
        approval.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "more than once"):
            review.render(self.audio, directory / "proposals.json", approval,
                          output, evidence)
        self.assertFalse(output.exists())
        data["approved"] = [dict(data["approved"][0]), {
            "id": result["candidates"][1]["id"],
            "reviewed_preview_sha256": result["candidates"][1]["preview_sha256"],
            "cut_start_seconds": .25, "cut_end_seconds": .3}]
        approval.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "overlap"):
            review.render(self.audio, directory / "proposals.json", approval,
                          output, evidence)
        self.assertFalse(output.exists())
        data["approved"] = [dict(data["approved"][0])]
        approval.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "separate from source"):
            review.render(self.audio, directory / "proposals.json", approval,
                          self.audio, evidence)
        self.assertEqual(review.sha256(self.audio), self.source_hash)

    def test_manual_flag_and_preview_integrity_are_required(self):
        self.run_report.write_text(json.dumps({"wav_sha256": self.source_hash,
                                               "changed": True}))
        with self.assertRaisesRegex(ValueError, "assembly and run report"):
            review.propose(self.audio, self.run_report, self.assembly,
                           self.root / "review", self.manual)
        self.run_report.write_text(json.dumps({"wav_sha256": self.source_hash}))
        manual = json.loads(self.manual.read_text())
        manual["audio_sha256"] = "0" * 64
        self.manual.write_text(json.dumps(manual))
        with self.assertRaisesRegex(ValueError, "manual filler"):
            review.propose(self.audio, self.run_report, self.assembly,
                           self.root / "review", self.manual)
        self.assertFalse((self.root / "review").exists())
        manual["audio_sha256"] = self.source_hash
        manual["flags"][0]["end_seconds"] = float("nan")
        self.manual.write_text(json.dumps(manual))
        with self.assertRaisesRegex(ValueError, "finite time"):
            review.propose(self.audio, self.run_report, self.assembly,
                           self.root / "review", self.manual)
        self.assertFalse((self.root / "review").exists())
        manual["flags"][0]["end_seconds"] = 1.05
        self.manual.write_text(json.dumps(manual))
        directory, result = self.proposals()
        approval = self.approval(directory, result)
        preview = directory / result["candidates"][0]["preview_file"]
        preview.write_bytes(b"changed preview")
        with self.assertRaisesRegex(ValueError, "preview"):
            review.render(self.audio, directory / "proposals.json", approval,
                          self.root / "edited.wav", self.root / "edited.json")
        self.assertFalse((self.root / "edited.wav").exists())


if __name__ == "__main__":
    unittest.main()
