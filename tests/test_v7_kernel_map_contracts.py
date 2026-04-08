#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_MAPS = ROOT / "version" / "v7" / "kernel_maps"
REGISTRY_PATH = KERNEL_MAPS / "KERNEL_REGISTRY.json"
BINDINGS_PATH = KERNEL_MAPS / "kernel_bindings.json"
EXCLUDED_MAP_FILES = {
    "KERNEL_REGISTRY.json",
    "KERNEL_SOURCES.json",
    "kernel_bindings.json",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a JSON object")
    return data


class TestV7KernelMapContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_json(REGISTRY_PATH)
        cls.bindings = _load_json(BINDINGS_PATH)
        cls.registry_by_id = {
            str(kernel.get("id")): kernel
            for kernel in cls.registry.get("kernels", [])
            if isinstance(kernel, dict) and kernel.get("id")
        }

    def test_registry_covers_kernel_map_files(self) -> None:
        missing = []
        mismatched_source = []
        for path in sorted(KERNEL_MAPS.glob("*.json")):
            if path.name in EXCLUDED_MAP_FILES:
                continue
            doc = _load_json(path)
            kernel_id = doc.get("id")
            if not kernel_id:
                continue
            entry = self.registry_by_id.get(str(kernel_id))
            if entry is None:
                missing.append(path.name)
                continue
            if entry.get("_source_file") != path.name:
                mismatched_source.append((path.name, entry.get("_source_file")))
        self.assertEqual(missing, [], f"kernel maps missing from registry: {missing}")
        self.assertEqual(
            mismatched_source,
            [],
            f"kernel registry source-file mismatches: {mismatched_source}",
        )

    def test_rmsnorm_binding_contract_stays_split(self) -> None:
        rmsnorm_forward = self.bindings.get("bindings", {}).get("rmsnorm_forward")
        rmsnorm_backward = self.bindings.get("bindings", {}).get("rmsnorm_backward")
        self.assertIsNotNone(rmsnorm_forward)
        self.assertIsNotNone(rmsnorm_backward)

        forward_sources = {
            str(param.get("name")): param.get("source")
            for param in rmsnorm_forward.get("params", [])
        }
        backward_sources = {
            str(param.get("name")): param.get("source")
            for param in rmsnorm_backward.get("params", [])
        }

        self.assertEqual(forward_sources.get("rstd_cache"), "null")
        self.assertEqual(backward_sources.get("rstd_cache"), "activation:rstd_cache")

        rmsnorm_forward_map = _load_json(KERNEL_MAPS / "rmsnorm_forward.json")
        output_names = {str(output.get("name")) for output in rmsnorm_forward_map.get("outputs", [])}
        self.assertIn("rstd", output_names)

    def test_exact_training_kernels_are_not_inference_eligible(self) -> None:
        expected_modes = {
            "geglu_forward_exact": {"inference": False, "training": True, "backward": False},
            "geglu_backward_exact": {"inference": False, "training": True, "backward": True},
            "swiglu_forward_exact": {"inference": False, "training": True, "backward": False},
            "swiglu_backward_exact": {"inference": False, "training": True, "backward": True},
        }
        for kernel_id, modes in expected_modes.items():
            with self.subTest(kernel_id=kernel_id):
                doc = _load_json(KERNEL_MAPS / f"{kernel_id}.json")
                self.assertEqual(doc.get("modes"), modes)
                self.assertIn(kernel_id, self.registry_by_id)
                self.assertIn(kernel_id, self.bindings.get("bindings", {}))


if __name__ == "__main__":
    unittest.main()
