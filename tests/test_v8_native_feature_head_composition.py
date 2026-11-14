"""Declared native feature edge into the complete small head; synthetic decoder."""

import ctypes as C
import tempfile
from pathlib import Path
import subprocess
import shutil
import unittest
import numpy as np
from tests.test_v8_decoder_feature_sequence import FAKE, ROOT
from tests.test_v8_clef_flash_complete_head import (
    synthetic_weights,
    build_circuit,
    effective_weights,
)
from tests.test_v8_clef_flash_head_ingress import compile_case
from tests.test_v8_clef_flash_evidence0 import prepare
from version.v8.scripts.compose_decoder_features_v8 import compose_decoder_features


class NativeFeatureHeadComposition(unittest.TestCase):
    def test_composition_is_explicit_and_rejects_duplicate_sources(self):
        graph = build_circuit(tokens=6, channels=8, width=8, heads=2, feedforward=16)
        import copy

        original = copy.deepcopy(graph)
        connected = compose_decoder_features(graph)
        self.assertEqual(graph, original)
        with self.assertRaises(ValueError):
            compose_decoder_features(connected)
        with self.assertRaises(ValueError):
            compose_decoder_features(graph, token_input="ids;bad")
        with self.assertRaises(ValueError):
            compose_decoder_features(graph, token_input="input_ids")
        graph["checked_native_entry"] = False
        with self.assertRaises(ValueError):
            compose_decoder_features(graph)

    def test_complete_head_receives_native_features_and_rejects_failed_component(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch needed to construct independent small head weights")
        graph = build_circuit(tokens=6, channels=8, width=8, heads=2, feedforward=16)
        weights = effective_weights(graph, synthetic_weights())
        connected = compose_decoder_features(graph)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture = (
                FAKE.replace("features[4]", "features[8]")
                .replace("c<4", "c<8")
                .replace("{1,4,features,4,", "{1,8,features,8,")
            )
            (folder / "fake.c").write_text(fixture)
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-I" + str(ROOT / "include"),
                    str(folder / "fake.c"),
                    "-o",
                    str(folder / "fake.so"),
                ],
                check=True,
            )
            component = C.CDLL(str(folder / "fake.so"))
            component.test_context.restype = C.c_void_p
            component.test_context_bytes.restype = C.c_size_t
            ids = np.array([1, 3, 2, 4, 6, 1], dtype="i4")
            hidden = np.array(
                [
                    [
                        t * (c + 1) + np.float32(0.1) * np.float32(i + 1)
                        for c in range(8)
                    ]
                    for i, t in enumerate(ids)
                ],
                dtype="f4",
            )
            baseline_arena, baseline, baseline_view, _, _ = compile_case(
                folder / "baseline", graph, weights
            )
            prepare(baseline_view, hidden)
            baseline_view("question_type_ids")[:] = [0, 2]
            self.assertEqual(baseline(baseline_arena, len(baseline_arena)), 0)
            expected = baseline_view("option_logits").copy()
            arena, execute, view, calls, _ = compile_case(
                folder / "connected",
                connected,
                weights,
                activation_dtypes={"backbone_ids": "i32"},
            )
            execute.argtypes = [
                C.POINTER(C.c_uint8),
                C.c_size_t,
                C.c_void_p,
                C.c_size_t,
            ]
            prepare(view, np.full_like(hidden, -900))
            view("question_type_ids")[:] = [0, 2]
            view("backbone_ids")[:] = ids
            component.test_mode(0)
            self.assertEqual(
                execute(
                    arena,
                    len(arena),
                    component.test_context(),
                    component.test_context_bytes(),
                ),
                0,
            )
            np.testing.assert_array_equal(view("hidden"), hidden)
            np.testing.assert_array_equal(view("option_logits"), expected)
            component.test_mode(1)
            view("option_logits")[:] = -901
            self.assertLess(
                execute(
                    arena,
                    len(arena),
                    component.test_context(),
                    component.test_context_bytes(),
                ),
                0,
            )
            np.testing.assert_array_equal(view("option_logits"), -901)
            component.test_mode(0)
            self.assertEqual(
                execute(
                    arena,
                    len(arena),
                    component.test_context(),
                    component.test_context_bytes(),
                ),
                0,
            )
            np.testing.assert_array_equal(view("option_logits"), expected)
            self.assertEqual(
                calls["operations"][0]["function"],
                "decoder_features_sequence_checked_f32",
            )
            # Deploy prepared weights/IDs/spans, not captured hidden features.
            # A new C-only process must run the declared component and head.
            request_inputs = {
                "backbone_ids",
                "input_ids",
                "question_spans",
                "option_spans",
                "global_span",
                "option_counts",
                "question_type_ids",
            }
            for name in connected["activation_buffers"]:
                if name not in request_inputs:
                    view(name)[:] = 0
            view("hidden")[:] = -903
            view("option_logits")[:] = -904
            host_source = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(void) {
    void *component=dlopen("./fake.so",RTLD_NOW|RTLD_LOCAL);
    void *head=dlopen("./generated.so",RTLD_NOW|RTLD_LOCAL);
    if(!component||!head)return 1;
    void *(*context)(void)=(void*(*)(void))dlsym(component,"test_context");
    size_t (*extent)(void)=(size_t(*)(void))dlsym(component,"test_context_bytes");
    int (*execute)(uint8_t*,size_t,const void*,size_t)=
      (int(*)(uint8_t*,size_t,const void*,size_t))dlsym(head,"ENTRY_NAME");
    if(!context||!extent||!execute)return 2;
    FILE *f=fopen("arena.bin","rb");
    if(!f||fseek(f,0,SEEK_END))return 3;
    long size=ftell(f);if(size<=0||size>1048576)return 4;rewind(f);
    uint8_t *arena=aligned_alloc(64,((size_t)size+63)&~(size_t)63);
    if(!arena||fread(arena,1,(size_t)size,f)!=(size_t)size)return 5;
    fclose(f);
    if(execute(arena,(size_t)size,context(),extent()))return 6;
    f=fopen("result.bin","wb");
    if(!f||fwrite(arena,1,(size_t)size,f)!=(size_t)size)return 7;
    fclose(f);free(arena);dlclose(head);dlclose(component);return 0;
}
""".replace("ENTRY_NAME", connected["native_entry"]["function"])
            deployed = folder / "deployed"
            deployed.mkdir()
            (deployed / "host.c").write_text(host_source)
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(deployed / "host.c"),
                    "-ldl",
                    "-o",
                    str(deployed / "host"),
                ],
                check=True,
            )
            (deployed / "arena.bin").write_bytes(bytes(arena))
            shutil.copyfile(
                folder / "connected/generated.so", deployed / "generated.so"
            )
            shutil.copyfile(folder / "fake.so", deployed / "fake.so")
            shutil.rmtree(folder / "connected")
            subprocess.run(
                ["./host"],
                cwd=deployed,
                env={"PATH": "/usr/bin:/bin", "CK_NUM_THREADS": "1"},
                check=True,
            )
            data = (deployed / "result.bin").read_bytes()
            self.assertEqual(len(data), len(arena))
            C.memmove(C.addressof(arena), data, len(data))
            np.testing.assert_array_equal(view("hidden"), hidden)
            np.testing.assert_array_equal(view("option_logits"), expected)
            (deployed / "fake.so").unlink()
            self.assertNotEqual(
                subprocess.run(
                    ["./host"], cwd=deployed, capture_output=True
                ).returncode,
                0,
            )
