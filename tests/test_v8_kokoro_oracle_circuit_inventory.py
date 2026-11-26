"""Check oracle regeneration without model weights or a native build."""

from importlib import import_module
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tests.v8_kokoro_oracle_circuit_support import SNAPSHOT_SHA256, oracle_circuit_bytes


ROOT = Path(__file__).resolve().parents[1]


class KokoroOracleCircuitInventoryTest(unittest.TestCase):
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
