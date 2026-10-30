"""One generated Qwen talker prefix handles different valid prompt rows."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import (
    build_circuit as fixed_prompt, prompt_plan,
)
from version.v8.tts.build_qwen3_tts_runtime_prompt_first_block_circuit import (
    build_circuit as runtime_prompt,
)
from version.v8.tts.qwen3_tts_pinned_session import (
    _compile, _prompt_ids, _view, _weight_shapes,
)


MODEL_DIR = os.environ.get("CKE_QWEN3_TTS_MODEL_DIR")
FIXTURE = (Path(__file__).resolve().parents[1] / "tests/fixtures/tts"
           / "qwen3_tts_0b6_customvoice/qwen3_tts_hi_first_block.npz")
ORACLE_MAX_ABS = {"block_input": 5e-7, "q_head": 5e-6,
                  "block_output": 1.2e-5}


def _load(graph, model_dir: Path, folder: Path):
    from safetensors import safe_open

    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as model, \
            safe_open(model_dir / "speech_tokenizer/model.safetensors",
                      framework="pt", device="cpu") as codec:
        shapes = _weight_shapes(graph, model, codec)
        acts, weights, layout, _, library = _compile(folder, graph, shapes)
        arena_bytes = layout["memory"]["arena"]["total_size"]
        raw = (ctypes.c_uint8 * (arena_bytes + 63))()
        arena = (ctypes.c_uint8 * arena_bytes).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for name, item in weights.items():
            value = np.ascontiguousarray(model.get_tensor(name).float().numpy())
            if value.shape != shapes[name] or value.nbytes != item["size"]:
                raise ValueError("weight disagrees with planned storage: " + name)
            arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
    native = ctypes.CDLL(str(library))
    run = getattr(native, graph["native_entry"]["function"])
    run.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    run.restype = ctypes.c_int
    return raw, arena, arena_bytes, acts, layout, run


def _execute(state, text_ids, codec_ids, *, valid_context: int | None = None):
    raw, arena, arena_bytes, acts, layout, run = state
    base = layout["memory"]["arena"]["activations_base"]
    ctypes.memset(ctypes.addressof(arena) + base, 0, arena_bytes - base)
    _view(arena, acts, "text_ids", np.int32, (len(text_ids),))[:] = text_ids
    _view(arena, acts, "codec_ids", np.int32, (7,))[:] = codec_ids
    plan = prompt_plan(len(text_ids) - 6)
    for name, indices in plan.items():
        _view(arena, acts, name, np.int32, (len(indices),))[:] = indices
    if valid_context is not None:
        _view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = valid_context
    status = run(arena, arena_bytes)
    rows = len(text_ids) + 5
    if valid_context is not None and status == 0:
        # Runtime extents are checked locals in generated C, not persistent
        # activation values. Verify their effects through outputs and padding.
        capacity_rows = acts["block_output"]["size"] // (1024 * 4)
        padding = _view(arena, acts, "block_output", np.float32,
                        (capacity_rows, 1024))[rows:]
        if np.any(padding != 0):
            raise AssertionError("runtime block output modified unused row capacity")
    outputs = {
        "block_input": _view(arena, acts, "block_input", np.float32,
                             (rows, 1024)).copy(),
        "q_head": _view(arena, acts, "q_head", np.float32,
                        (16, rows, 128)).copy(),
        "block_output": _view(arena, acts, "block_output", np.float32,
                              (rows, 1024)).copy(),
    }
    return status, outputs


class Qwen3TTSRuntimePromptFirstBlockTest(unittest.TestCase):
    def test_committed_circuit_matches_authoring(self):
        circuit = (Path(__file__).resolve().parents[1] / "version/v8/circuits"
                   / "qwen3_tts_runtime_prompt_first_block_bounded.json")
        self.assertEqual(json.loads(circuit.read_text()), runtime_prompt())

    def test_short_reference_integrity(self):
        manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
        baseline = json.loads((FIXTURE.parent /
                               "qwen3_tts_customvoice_reference.json").read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        self.assertEqual((manifest["text"], manifest["context_tokens"],
                          manifest["prompt_rows"]), ("Hi.", 2, 13))
        self.assertEqual(manifest["model_files_sha256"],
                         baseline["model"]["files_sha256"])
        self.assertEqual(manifest["reference_files_sha256"],
                         baseline["reference"]["files_sha256"])
        with np.load(FIXTURE) as capture:
            for name, shape in (("block_input", (13, 1024)),
                                ("q_head", (16, 13, 128)),
                                ("block_output", (13, 1024))):
                self.assertEqual(capture[name].shape, shape)
                self.assertTrue(np.isfinite(capture[name]).all())

    def test_declared_row_bindings_and_boundaries(self):
        for invalid in (0, -1, 118, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime_prompt(context_capacity=invalid)
        graph = runtime_prompt(context_capacity=7)
        self.assertEqual(graph["runtime_lengths"]["valid_prompt_rows"]["capacity"], 18)
        self.assertEqual(graph["block_types"]["first_block"]["body"]["ops"][0]
                         ["runtime_scalar_bindings"], {"rows": "valid_prompt_rows"})
        self.assertEqual(len(graph["block_types"]["text_projection"]["header"]), 7)

    @unittest.skipUnless(MODEL_DIR, "pinned Qwen model unavailable; runtime row parity NOT_TESTED")
    def test_two_lengths_and_recovery_on_one_generated_entry(self):
        from transformers import AutoTokenizer

        model_dir = Path(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        requests = (("Hi.", 2), ("Hello, this is CKE.", 7))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            runtime = _load(runtime_prompt(context_capacity=7), model_dir,
                            folder / "runtime")
            fixed = {
                count: _load(fixed_prompt(count), model_dir, folder / f"fixed{count}")
                for _, count in requests
            }
            results = {}
            with np.load(FIXTURE) as capture:
                short_reference = {name: capture[name].copy()
                                   for name in ORACLE_MAX_ABS}
            for text, count in (requests[0], requests[1], requests[0]):
                text_ids, codec_ids = _prompt_ids(
                    model_dir, text, tokenizer, context_tokens=count)
                dynamic_status, dynamic = _execute(
                    runtime, text_ids, codec_ids, valid_context=count)
                fixed_status, expected = _execute(
                    fixed[count], text_ids, codec_ids)
                self.assertEqual((dynamic_status, fixed_status), (0, 0))
                for name in expected:
                    with self.subTest(text=text, checkpoint=name):
                        np.testing.assert_allclose(dynamic[name], expected[name],
                                                   rtol=1e-6, atol=2e-6)
                    if count == 2:
                        delta = (dynamic[name].astype(np.float64) -
                                 short_reference[name].astype(np.float64))
                        self.assertLessEqual(float(np.max(np.abs(delta))),
                                             ORACLE_MAX_ABS[name])
                if text in results:
                    for name in dynamic:
                        np.testing.assert_array_equal(dynamic[name], results[text][name])
                results[text] = dynamic
            raw, arena, arena_bytes, acts, layout, run = runtime
            for invalid in (0, 8):
                base = layout["memory"]["arena"]["activations_base"]
                ctypes.memset(ctypes.addressof(arena) + base, 0, arena_bytes - base)
                _view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = invalid
                output = _view(arena, acts, "block_output", np.float32, (18, 1024))
                output.fill(-901)
                self.assertNotEqual(run(arena, arena_bytes), 0)
                np.testing.assert_array_equal(output, -901)
