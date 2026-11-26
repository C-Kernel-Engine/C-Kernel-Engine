"""Check Kokoro diagnostic regeneration without model weights or a native build."""

from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.v8_kokoro_oracle_circuit_support import SNAPSHOT_SHA256, oracle_circuit_bytes
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit


ROOT = Path(__file__).resolve().parents[1]
STAGE_SNAPSHOTS = ROOT / "tests/fixtures/tts/kokoro_stage_circuit_snapshots.json"
CHECKED_IN = {
    "kokoro_phoneme_encoder_bounded.json",
    "kokoro_waveform_bounded.json",
    "kokoro_runtime_waveform_bounded.json",
}


class KokoroOracleCircuitInventoryTest(unittest.TestCase):
    def test_only_waveform_and_encoder_references_are_checked_in(self):
        actual = {path.name for path in (ROOT / "version/v8/circuits").glob("kokoro*.json")}
        self.assertEqual(actual, CHECKED_IN)

    def test_stage_builders_reproduce_previous_snapshots(self):
        snapshots = json.loads(STAGE_SNAPSHOTS.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory, patch.object(
                sys, "path", [str(ROOT / "version/v8/tts"), *sys.path]):
            for builder, snapshot in snapshots.items():
                with self.subTest(builder=builder):
                    author = import_module(builder)
                    circuit = materialize_stage_circuit(Path(directory), author)
                    self.assertEqual(circuit.name, snapshot["file"])
                    self.assertEqual(sha256(circuit.read_bytes()).hexdigest(),
                                     snapshot["sha256"])
                    self.assertEqual(author.OUTPUT.parent, ROOT / "build/kokoro-diagnostics")

    def test_oracle_builders_reproduce_previous_snapshots(self):
        with patch.object(sys, "path", [str(ROOT / "version/v8/tts"), *sys.path]):
            for name in SNAPSHOT_SHA256:
                with self.subTest(circuit=name):
                    stage = name.removeprefix("kokoro_").removesuffix("_oracle_input_bounded")
                    author = import_module(f"build_kokoro_{stage}_oracle_circuit")
                    graph = author.build_circuit()
                    self.assertEqual(graph["name"], name)
                    oracle_circuit_bytes(graph)
                    self.assertEqual(author.OUTPUT.parent, ROOT / "build/kokoro-diagnostics")
                    self.assertFalse((ROOT / "version/v8/circuits" / (name + ".json")).exists())
