"""Generic native decoder sequence contract; no Python token scheduler."""

import ctypes as C
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAKE = r"""
#include "ck_decoder_features.h"
#include <math.h>
static float features[4];
static int position, mode, calls, cancelled;
static int decode(int32_t token, float *unused) {
    (void)unused;
    ++calls; ++position;
    for(int c=0;c<4;++c) features[c]=(float)(token*(c+1))+0.1f*(float)position;
    if(mode==1 && position==2) return -1;
    if(mode==2) ++position;
    if(mode==3) features[1]=NAN;
    if(mode==5 && position==1) cancelled=1;
    if(mode==6 && position==3) cancelled=1;
    return 0;
}
static void reset(void) { if(mode!=4) position=0; }
static int active(void) { return position; }
static int capacity(void) { return 8; }
static int vocabulary(void) { return 10; }
static int should_cancel(void *unused) { (void)unused;return cancelled; }
static CKDecoderFeatures context={1,4,features,4,decode,reset,active,capacity,vocabulary,should_cancel,NULL};
void *test_context(void) { return &context; }
size_t test_context_bytes(void) { return sizeof(context); }
void test_mode(int value) { mode=value;position=value==4?3:0;calls=0;cancelled=0; }
int test_calls(void) { return calls; }
int test_active(void) { return position; }
float *test_features(void) { return features; }
"""


def generated_fixture():
    graph = dict(
        version=3,
        name="generic_component_feature_fixture",
        family="bounded_graph",
        checked_native_entry=True,
        contract={"runtime_invariants": {"inference_only": True}},
        activation_buffers={"ids": {"shape": [2]}, "features": {"shape": [2, 4]}},
        activation_bindings={"ids": "ids", "features": "features"},
        native_entry=dict(
            function="ck_collect_features",
            params=[
                dict(c_type="uint8_t *", name="arena"),
                dict(c_type="size_t", name="arena_bytes"),
                dict(c_type="const void *", name="decoder_context"),
                dict(c_type="size_t", name="decoder_context_bytes"),
            ],
            arena=dict(pointer="arena", bytes="arena_bytes"),
        ),
        sequence=["collect"],
        block_types={
            "collect": {
                "body": {
                    "ops": [
                        dict(
                            id="collect_features",
                            op="decoder_features_sequence",
                            kernel="decoder_features_sequence_checked_f32",
                            returns_status=True,
                            graph_slots=dict(
                                inputs={"ids": "external:ids"},
                                outputs={"output": "features"},
                            ),
                            params=dict(
                                T=2,
                                C=4,
                                Q=8,
                                call_constants=dict(
                                    id_elements=2,
                                    feature_tokens=2,
                                    feature_channels=4,
                                    feature_elements=8,
                                ),
                            ),
                        )
                    ]
                }
            }
        },
    )
    source = dict(
        config=dict(
            model=graph["name"],
            arch=graph["name"],
            num_layers=1,
            embed_dim=4,
            num_heads=1,
            num_kv_heads=1,
            head_dim=4,
            intermediate_size=4,
            context_length=2,
            max_seq_len=2,
            vocab_size=10,
            activation_buffer_dtypes={"ids": "i32"},
        ),
        entries=[],
        quant_summary={},
        template=graph,
    )
    return graph, source


class DecoderFeatureSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        (folder / "fake.c").write_text(FAKE)
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-shared",
                "-fPIC",
                "-I" + str(ROOT / "include"),
                str(ROOT / "src/runtime/decoder_features_sequence.c"),
                str(folder / "fake.c"),
                "-o",
                str(folder / "test.so"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cls.lib = C.CDLL(str(folder / "test.so"))
        cls.lib.test_context.restype = C.c_void_p
        cls.lib.test_context_bytes.restype = C.c_size_t
        cls.lib.test_features.restype = C.c_void_p
        cls.run_native = cls.lib.decoder_features_sequence_checked_f32
        cls.run_native.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
        ]
        cls.run_native.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def request(self, values, **changes):
        ids = np.asarray(values, dtype="i4")
        output = np.full((len(ids) * 4 + 4,), -991, dtype="f4")
        scratch = np.full((len(ids) * 4 + 4,), -992, dtype="f4")
        args = dict(
            context=self.lib.test_context(),
            context_bytes=self.lib.test_context_bytes(),
            ids=ids.ctypes.data,
            id_elements=ids.size,
            tokens=ids.size,
            channels=4,
            output=output.ctypes.data,
            output_elements=output.size,
            scratch=scratch.ctypes.data,
            scratch_elements=scratch.size,
        )
        args.update(changes)
        status = self.run_native(*args.values())
        return status, output, scratch

    def test_generated_entry_executes_declared_component(self):
        from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

        graph, source = generated_fixture()
        with tempfile.TemporaryDirectory() as directory:
            layout, calls, _, library, declared_fn = compile_native_graph(
                Path(directory), source, graph
            )
            self.assertEqual(
                declared_fn.argtypes,
                [C.POINTER(C.c_uint8), C.c_size_t, C.c_void_p, C.c_size_t],
            )
            from copy import deepcopy
            from version.v8.scripts.codegen_checked_calls_v8 import (
                emit_checked_calls,
                CheckedCallCodegenError,
            )

            altered = deepcopy(calls)
            altered["operations"][0]["args"][0]["expr"] = "(void*)0"
            with self.assertRaises(CheckedCallCodegenError):
                emit_checked_calls(altered, ROOT)
            altered = deepcopy(calls)
            altered["entry"]["params"][2]["c_type"] = "size_t"
            with self.assertRaises(CheckedCallCodegenError):
                emit_checked_calls(altered, ROOT)
            fn = library.ck_collect_features
            fn.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
            fn.restype = C.c_int
            size = layout["memory"]["arena"]["total_size"]
            storage = np.zeros(size + 64, dtype="u1")
            shift = (-storage.ctypes.data) % 64
            arena = storage[shift : shift + size]
            buffers = {b["name"]: b for b in layout["memory"]["activations"]["buffers"]}
            ids = np.ndarray(
                (2,), dtype="i4", buffer=arena, offset=buffers["ids"]["abs_offset"]
            )
            result = np.ndarray(
                (8,), dtype="f4", buffer=arena, offset=buffers["features"]["abs_offset"]
            )
            ids[:] = [1, 3]
            result[:] = -991
            self.lib.test_mode(0)
            self.assertEqual(
                fn(
                    arena.ctypes.data,
                    size,
                    self.lib.test_context(),
                    self.lib.test_context_bytes(),
                ),
                0,
            )
            expected = self.request([1, 3])[1][:-4]
            np.testing.assert_array_equal(result, expected)
            result[:] = -991
            self.lib.test_mode(1)
            self.assertLess(
                fn(
                    arena.ctypes.data,
                    size,
                    self.lib.test_context(),
                    self.lib.test_context_bytes(),
                ),
                0,
            )
            np.testing.assert_array_equal(result, -991)
            self.assertEqual(
                calls["operations"][0]["call_abi"]["kernel_id"],
                "decoder_features_sequence_checked_f32",
            )

    def test_native_sequence_replay_and_padding(self):
        self.lib.test_mode(0)
        first = None
        for values in ([1, 3], [2, 4, 6, 8, 3, 1, 0], [1, 3]):
            status, output, scratch = self.request(values)
            self.assertEqual(status, 0)
            expected = np.array(
                [
                    [
                        token * (c + 1) + np.float32(0.1) * np.float32(row + 1)
                        for c in range(4)
                    ]
                    for row, token in enumerate(values)
                ],
                dtype="f4",
            ).ravel()
            np.testing.assert_array_equal(output[:-4], expected)
            np.testing.assert_array_equal(output[-4:], -991)
            np.testing.assert_array_equal(scratch[-4:], -992)
            self.assertEqual(self.lib.test_active(), 0)
            if len(values) == 2:
                if first is None:
                    first = output.copy()
                else:
                    np.testing.assert_array_equal(first, output)

    def test_execution_failure_resets_without_publication_and_recovers(self):
        for mode in (1, 2, 3, 4, 5, 6):
            self.lib.test_mode(mode)
            status, output, _ = self.request([1, 2, 3])
            self.assertLess(status, 0)
            np.testing.assert_array_equal(output, -991)
            if mode != 4:
                self.assertEqual(self.lib.test_active(), 0)
            self.lib.test_mode(0)
            self.assertEqual(self.request([3, 1])[0], 0)

    def test_rejection_precedes_decode(self):
        for changes in (
            {"context": None},
            {"context_bytes": 0},
            {"channels": 3},
            {"id_elements": 1},
            {"output_elements": 7},
            {"scratch_elements": 7},
            {"tokens": 0},
            {"tokens": 2**63},
            {"ids": None},
            {"output": C.c_void_p(-4).value},
            {"output": self.lib.test_features()},
            {"scratch": self.lib.test_context()},
        ):
            self.lib.test_mode(0)
            status, output, scratch = self.request([1, 2], **changes)
            self.assertLess(status, 0)
            self.assertEqual(self.lib.test_calls(), 0)
            np.testing.assert_array_equal(output, -991)
            np.testing.assert_array_equal(scratch, -992)
        for ids in ([-1, 2], [1, 10], list(range(9))):
            self.lib.test_mode(0)
            self.assertLess(self.request(ids)[0], 0)
            self.assertEqual(self.lib.test_calls(), 0)


if __name__ == "__main__":
    unittest.main()
