from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/build_training_capability_ledger_v8.py"
REPORT = ROOT / "docs/site/_pages/v8-training-capability-ledger.html"


class TrainingCapabilityLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("training_capability_ledger_v8_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.ledger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ledger)

    def test_ledger_is_derived_and_does_not_claim_execution(self) -> None:
        document = self.ledger.inventory()
        self.assertGreater(document["provider_count"], 100)
        self.assertEqual(document["execution_evidence"], "NOT_MEASURED_BY_MAP_INVENTORY")
        self.assertEqual(REPORT.read_text(encoding="utf-8"), self.ledger.render_html(document))
        self.assertTrue(any(row["declared_backward"] for row in document["providers"] if row["family"] == "fp32-tagged"))
        self.assertFalse(any(row["declared_backward"] for row in document["providers"] if row["family"] == "quantized"))

    def test_dtype_tags_and_missing_backward_mode_are_distinct(self) -> None:
        family = self.ledger._family
        self.assertEqual(family({"id": "gemm_q4_k", "quant": {"weight": "q4_k"}}, Path("gemm_q4_k.json")), "quantized")
        self.assertEqual(family({"id": "rmsnorm_bf16_storage", "quant": {"output": "fp32"}}, Path("rmsnorm_bf16_storage.json")), "bf16-tagged")
        self.assertEqual(family({"id": "gemm_backward_f32", "quant": {"output": "fp32"}}, Path("gemm_backward_f32.json")), "fp32-tagged")
        document = self.ledger.inventory()
        uncertain = [row for row in document["providers"] if row["backward_named_without_mode"]]
        self.assertTrue(uncertain)
        self.assertTrue(all(not row["declared_backward"] for row in uncertain))

    def test_authoring_and_map_operations_do_not_infer_compiler_certification(self) -> None:
        document = self.ledger.inventory()
        modules = {row["name"]: row for row in document["authoring_modules"]}
        self.assertIn("Linear", modules)
        self.assertIn("RMSNorm", modules)
        self.assertIn("TransformerBlock", modules)
        self.assertEqual(modules["Linear"]["lowering_evidence"], "NOT_ASSESSED_BY_MAP_INVENTORY")
        self.assertEqual(modules["Linear"]["backward_evidence"], "NOT_ASSESSED_BY_MAP_INVENTORY")
        self.assertTrue(document["authoring_source_sha256"])
        self.assertTrue(document["operations"])
        for operation in document["operations"]:
            self.assertEqual(operation["semantic_operation"], "NOT_MAPPED_BY_MAP_INVENTORY")
            self.assertEqual(operation["authoring_module"], "NOT_MAPPED_BY_MAP_INVENTORY")
            self.assertEqual(operation["generated_training_evidence"], "NOT_MEASURED_BY_MAP_INVENTORY")
            matching = [row for row in document["providers"]
                        if (row["family"], row["op"]) == (operation["family"], operation["map_op"])]
            self.assertEqual(operation["provider_ids"], [row["id"] for row in matching])
            self.assertEqual(operation["declared_backward_ids"],
                             [row["id"] for row in matching if row["declared_backward"]])


if __name__ == "__main__":
    unittest.main()
