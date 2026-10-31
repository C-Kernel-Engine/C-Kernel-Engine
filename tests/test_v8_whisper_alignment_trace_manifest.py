"""Opt-in Whisper trace authoring and token correspondence contracts."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version/v8/scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


author = load("build_whisper_alignment_trace_manifest_v8")
aligner = load("align_whisper_trace_v8")


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join({1: "Hello", 2: " world", 3: " again"}[item] for item in ids)


class WhisperAlignmentTraceManifestTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "config": {"num_heads": 3, "num_layers": 2,
                       "encoder_memory_length": 7, "head_dim": 4},
            "template": {"block_types": {"decoder": {"body": {"ops": [
                "self_attn", "cross_attn", "residual"
            ]}}}},
        }
        self.generation = {"alignment_heads": [[0, 1], [1, 2]]}

    def test_decode_observer_and_prefill_layout_are_distinct(self):
        original = copy.deepcopy(self.source)
        decode = author.trace_manifest(self.source, self.generation)
        prefill = author.trace_manifest(self.source, self.generation, emit_observer=False)
        self.assertEqual(self.source, original)
        operations = decode["template"]["block_types"]["decoder"]["body"]["ops"]
        self.assertEqual(operations[2]["op"], "attention_last_query_prob_trace")
        self.assertEqual(operations[1], "cross_attn")
        self.assertEqual(prefill["template"]["block_types"]["decoder"]["body"]["ops"],
                         original["template"]["block_types"]["decoder"]["body"]["ops"])
        self.assertEqual(decode["template"]["activation_buffers"],
                         prefill["template"]["activation_buffers"])
        self.assertEqual(decode["config"]["alignment_trace_call_constants"]["trace_elements"], 42)

    def test_rejects_wrong_head_selection_and_missing_edge(self):
        for heads in ([], [[0, 1], [0, 1]], [[2, 1]], [[0, 3]], [[True, 1]], [[0]]):
            with self.subTest(heads=heads):
                with self.assertRaises(ValueError):
                    author.trace_manifest(self.source, {"alignment_heads": heads})
        source = copy.deepcopy(self.source)
        source["template"]["block_types"]["decoder"]["body"]["ops"].remove("cross_attn")
        with self.assertRaises(ValueError):
            author.trace_manifest(source, self.generation)

    def test_word_correspondence_preserves_token_and_character_spans(self):
        text, words = aligner._word_tokens(FakeTokenizer(), [1, 2, 3])
        self.assertEqual(text, "Hello world again")
        self.assertEqual([(w["text"], w["token_start"], w["token_end_exclusive"])
                          for w in words],
                         [("Hello", 0, 1), ("world", 1, 2), ("again", 2, 3)])

    def test_trace_identity_and_normalization_rejected_before_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            attention = np.ones((1, 2, 3), np.float32) / 3
            np.savez_compressed(path, selected_attention=attention,
                                token_ids=np.asarray([1, 2], np.int32),
                                alignment_heads=np.asarray([[0, 1]], np.int32),
                                frame_capacity=np.asarray(3, np.int32))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            report = {"encoder": {}, "decoder": {
                "generated_tokens": [1, 2],
                "attention_trace": {"sha256": sha, "selected_heads": [[0, 1]],
                                    "shape": [1, 2, 3]},
            }}
            with self.assertRaisesRegex(ValueError, "hash"):
                broken = copy.deepcopy(report)
                broken["decoder"]["attention_trace"]["sha256"] = "0" * 64
                aligner.align_trace(broken, path, Path(directory), Path(directory),
                                    Path(directory) / "missing.so", 0)
            with self.assertRaisesRegex(ValueError, "geometry"):
                broken = copy.deepcopy(report)
                broken["decoder"]["attention_trace"]["shape"] = [1, 3, 2]
                aligner.align_trace(broken, path, Path(directory), Path(directory),
                                    Path(directory) / "missing.so", 0)


if __name__ == "__main__":
    unittest.main()
