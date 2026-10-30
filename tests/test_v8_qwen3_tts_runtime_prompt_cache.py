"""Checked Qwen prompt rows seed caller-owned KV state through generated C."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_runtime_prompt_first_block import _load
from version.v8.tts.build_qwen3_tts_runtime_prompt_cache_circuit import (
    build_circuit,
)
from version.v8.tts.qwen3_tts_pinned_session import _prompt_ids, _view
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"
           / "qwen3_tts_hi_cache_rows.npz")
MODEL_DIR = os.environ.get("CKE_QWEN3_TTS_MODEL_DIR")


def execute(state, text_ids, codec_ids, context_tokens):
    raw, arena, arena_bytes, acts, layout, run = state
    base = layout["memory"]["arena"]["activations_base"]
    ctypes.memset(ctypes.addressof(arena) + base, 0, arena_bytes - base)
    _view(arena, acts, "text_ids", np.int32, (len(text_ids),))[:] = text_ids
    _view(arena, acts, "codec_ids", np.int32, (7,))[:] = codec_ids
    for name, indices in prompt_plan(context_tokens).items():
        _view(arena, acts, name, np.int32, (len(indices),))[:] = indices
    _view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = context_tokens
    status = run(arena, arena_bytes)
    rows = context_tokens + 11
    result = {}
    for layer in range(28):
        for kind in ("k", "v"):
            name = f"talker_layer{layer}_cache_{kind}"
            capacity = acts[name]["size"] // (8 * 128 * 4)
            cache = _view(arena, acts, name, np.float32, (8, capacity, 128))
            result[layer, kind] = cache.copy()
    return status, rows, result


class Qwen3TTSRuntimePromptCacheTest(unittest.TestCase):
    def test_circuit_regeneration_and_scalar_binding(self):
        graph = build_circuit()
        committed = (ROOT / "version/v8/circuits"
                     / "qwen3_tts_runtime_prompt_cache_bounded.json")
        self.assertEqual(json.loads(committed.read_text()), graph)
        seeds = graph["block_types"]["talker_cache_seed"]["body"]["ops"]
        self.assertEqual(len(seeds), 28)
        first = graph["block_types"]["first_block"]["body"]["ops"]
        for layer in range(1, 28):
            repeated = graph["block_types"][f"talker_layer_{layer}"]["body"]["ops"]
            self.assertEqual(len(repeated), len(first))
            for original, current in zip(first, repeated):
                self.assertEqual(current["runtime_scalar_bindings"],
                                 original["runtime_scalar_bindings"])
                for argument in current["runtime_scalar_bindings"]:
                    self.assertNotIn(argument, current["params"]["call_constants"])
        for head in graph["block_types"]["talker_head"]["body"]["ops"]:
            self.assertEqual(head["consumes_runtime_lengths"], ["valid_prompt_rows"])
        for seed in seeds:
            self.assertEqual(seed["runtime_scalar_bindings"],
                             {"tokens": "valid_prompt_rows"})
            self.assertNotIn("tokens", seed["params"]["call_constants"])
        for invalid in (0, 29, True):
            with self.subTest(layers=invalid), self.assertRaises(ValueError):
                build_circuit(layers=invalid)
        for invalid in (18, 0, -1, True, 257):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_circuit(cache_capacity=invalid)

    def test_short_reference_integrity(self):
        manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
        prefix = json.loads((FIXTURE.parent /
                             "qwen3_tts_hi_first_block.json").read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        for identity in ("model_files_sha256", "reference_files_sha256"):
            self.assertEqual(manifest[identity], prefix[identity])
        with np.load(FIXTURE) as capture:
            for name in ("k_head", "v_head"):
                self.assertEqual(capture[name].shape, (8, 13, 128))
                self.assertTrue(np.isfinite(capture[name]).all())

    @unittest.skipUnless(MODEL_DIR, "pinned Qwen assets unavailable; cache parity NOT_TESTED")
    def test_two_lengths_seed_cache_and_recover(self):
        from transformers import AutoTokenizer

        model_dir = Path(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        with tempfile.TemporaryDirectory() as temp:
            state = _load(build_circuit(), model_dir, Path(temp) / "runtime")
            _, arena, arena_bytes, acts, layout, run = state
            with np.load(FIXTURE) as capture:
                short_reference = {kind: capture[f"{kind}_head"].copy()
                                   for kind in ("k", "v")}
            full_oracle_path = (FIXTURE.parent / "qwen3_tts_talker_prefill.npz")
            with np.load(full_oracle_path) as capture:
                long_logits = capture["logits"][0].copy()
            first_short = None
            for text, count in (("Hi.", 2), ("Hello, this is CKE.", 7), ("Hi.", 2)):
                text_ids, codec_ids = _prompt_ids(
                    model_dir, text, tokenizer, context_tokens=count)
                status, rows, caches = execute(state, text_ids, codec_ids, count)
                self.assertEqual(status, 0)
                if count == 7:
                    logits = _view(arena, acts, "talker_logits", np.float32,
                                   (18, 3072))
                    self.assertLessEqual(float(np.max(np.abs(logits - long_logits))),
                                         1.5e-4)
                for (layer, kind), cache in caches.items():
                    source_name = (f"layer{layer}_" if layer else "") + f"{kind}_head"
                    source = _view(arena, acts, source_name, np.float32,
                                   (8, rows, 128))
                    np.testing.assert_array_equal(cache[:, :rows], source)
                    np.testing.assert_array_equal(cache[:, rows:], 0)
                    if count == 2 and layer == 0:
                        error = np.max(np.abs(cache[:, :rows] -
                                              short_reference[kind]))
                        self.assertLessEqual(float(error), 8e-5)
                    if count == 2 and first_short is not None:
                        np.testing.assert_array_equal(cache, first_short[layer, kind])
                if first_short is None:
                    first_short = caches
            base = layout["memory"]["arena"]["activations_base"]
            for invalid in (0, 8):
                ctypes.memset(ctypes.addressof(arena) + base, 0, arena_bytes - base)
                _view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = invalid
                for layer in range(28):
                    for kind in ("k", "v"):
                        _view(arena, acts, f"talker_layer{layer}_cache_{kind}",
                              np.float32, (8, 21, 128)).fill(-901)
                self.assertNotEqual(run(arena, arena_bytes), 0)
                for layer in range(28):
                    for kind in ("k", "v"):
                        cache = _view(arena, acts, f"talker_layer{layer}_cache_{kind}",
                                      np.float32, (8, 21, 128))
                        np.testing.assert_array_equal(cache, -901)
                text_ids, codec_ids = _prompt_ids(
                    model_dir, "Hi.", tokenizer, context_tokens=2)
                status, rows, recovered = execute(state, text_ids, codec_ids, 2)
                self.assertEqual((status, rows), (0, 13))
                for key, expected in first_short.items():
                    np.testing.assert_array_equal(recovered[key], expected)
