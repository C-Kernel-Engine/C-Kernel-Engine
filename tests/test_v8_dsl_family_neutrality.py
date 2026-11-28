"""A resolved graph's execution must be invariant under a family rename."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.test_v8_runtime_extent_lowering import manifest
from version.v8.scripts import compile_checked_bundle_v8 as builder


def build(folder, label=None):
    _, source = manifest()
    graph = source["template"]
    if label:
        source["config"].update(model=label, arch=label)
        graph.update(name=label, family=label)
    folder.mkdir()
    layout, calls, library, _, _ = builder.compile_native_graph(folder, source, graph)
    # Retain all executable fields, including ABI, offsets, order and config.
    for document in (layout, calls):
        for field in ("model", "arch"):
            del document["config"][field]
    return (layout, calls, (folder / "generated.c").read_bytes()), library


def run_native(layout, library):
    folder = library.parent / "deploy"
    folder.mkdir()
    shutil.copy2(library, folder / library.name)
    buffers = {b["name"]: b["abs_offset"] for b in layout["memory"]["activations"]["buffers"]}
    args = [f"-DARENA_BYTES={layout['memory']['arena']['total_size']}"]
    for name in ("runtime_values", "audio_features", "runtime_valid_copy"):
        args.append(f"-D{name.upper()}={buffers[name]}")
    executable = folder / "probe"
    subprocess.run(["cc", "-std=c11", *args, "-x", "c", "-", "-x", "none",
                    "./" + library.name, "-o", executable.name],
                   input=NATIVE_PROBE, text=True, cwd=folder, check=True)
    subprocess.run([executable], cwd=folder, env={}, check=True)


NATIVE_PROBE = """
#include <stdint.h>
#include <string.h>
extern int ck_test_planned_graph(uint8_t *, size_t, int32_t *);
static _Alignas(64) uint8_t arena[ARENA_BYTES];

int main(void) {
    int32_t frames = -1;
    const int32_t bad[] = {3, 3, 3}, good[] = {1, 2, 1};
    const float features[] = {10, 20, 30, -1, 40, 50, 60, -1};
    const float expected[] = {10, 20, 20, 30};
    memcpy(arena + RUNTIME_VALUES, bad, sizeof bad);
    if (ck_test_planned_graph(arena, sizeof arena, &frames) != -2 || frames != -1) return 1;
    memcpy(arena + RUNTIME_VALUES, good, sizeof good);
    memcpy(arena + AUDIO_FEATURES, features, sizeof features);
    if (ck_test_planned_graph(arena, sizeof arena, &frames) || frames != 4) return 2;
    return memcmp(arena + RUNTIME_VALID_COPY, expected, sizeof expected) != 0;
}
"""


class DSLFamilyNeutrality(unittest.TestCase):
    def test_renamed_execution_and_native_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline, lib = build(root / "original")
            run_native(baseline[0], lib)
            for label in ("inkling", "unseen_robot"):
                with self.subTest(label=label):
                    renamed, other = build(root / label, label)
                    self.assertEqual(baseline, renamed)
                    run_native(renamed[0], other)

    def test_detects_family_keyed_provider_substitution(self):
        lower = builder.build_ir_v8.build_ir1_direct

        def injected(source, circuit, mode):
            if source["config"]["arch"] == "unseen_robot":
                expand = circuit["block_types"]["bounded_graph"]["body"]["ops"][0]
                expand["kernel"] = "audio_duration_expand_token_major_f32"
            return lower(source, circuit, mode=mode)

        with tempfile.TemporaryDirectory() as temp, patch.object(
            builder.build_ir_v8, "build_ir1_direct", side_effect=injected
        ):
            root = Path(temp)
            baseline, _ = build(root / "original")
            renamed, _ = build(root / "renamed", "unseen_robot")
            self.assertEqual(baseline[1]["operations"][1]["function"],
                             "audio_duration_expand_channel_major_f32")
            self.assertEqual(renamed[1]["operations"][1]["function"],
                             "audio_duration_expand_token_major_f32")
            self.assertNotEqual(baseline, renamed)
