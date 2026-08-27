import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "benchmarks" / "profile_v8_prefill_ops.py"
CODEGEN = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"


def _load_profile_module():
    spec = importlib.util.spec_from_file_location("profile_v8_prefill_ops", PROFILE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V8ProfileOccupancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _load_profile_module()

    def test_codegen_records_process_cpu_time(self) -> None:
        source = CODEGEN.read_text(encoding="utf-8")
        self.assertIn("CLOCK_PROCESS_CPUTIME_ID", source)
        self.assertIn("cpu_time_us", source)
        self.assertIn(
            "mode,kernel,op,layer,start_us,end_us,time_us,cpu_time_us,token_id",
            source,
        )

    def test_summary_reports_layer_worker_utilization(self) -> None:
        rows = [
            {
                "mode": "prefill",
                "kernel": "recurrent_kernel",
                "op": "recurrent_core",
                "layer": 0,
                "time_us": 100.0,
                "cpu_time_us": 1500.0,
            },
            {
                "mode": "prefill",
                "kernel": "attention_kernel",
                "op": "attention",
                "layer": 3,
                "time_us": 200.0,
                "cpu_time_us": 2400.0,
            },
        ]
        summary = self.profile._summarize(rows, limit=10, threads=16)

        self.assertAlmostEqual(summary["prefill_core_equivalents"], 13.0)
        self.assertAlmostEqual(summary["prefill_worker_utilization_pct"], 81.25)
        by_layer = {row["layer"]: row for row in summary["by_layer"]}
        self.assertAlmostEqual(by_layer[0]["core_equivalents"], 15.0)
        self.assertAlmostEqual(by_layer[0]["worker_utilization_pct"], 93.75)
        self.assertAlmostEqual(by_layer[3]["core_equivalents"], 12.0)
        self.assertAlmostEqual(by_layer[3]["worker_utilization_pct"], 75.0)

    def test_summary_attributes_unprofiled_transition_gaps(self) -> None:
        rows = [
            {
                "mode": "prefill", "kernel": "a", "op": "attention", "layer": 3,
                "start_us": 10.0, "end_us": 110.0,
                "time_us": 100.0, "cpu_time_us": 1600.0,
            },
            {
                "mode": "prefill", "kernel": "b", "op": "post_norm", "layer": 3,
                "start_us": 140.0, "end_us": 190.0,
                "time_us": 50.0, "cpu_time_us": 400.0,
            },
            {
                "mode": "prefill", "kernel": "c", "op": "input_norm", "layer": 4,
                "start_us": 250.0, "end_us": 270.0,
                "time_us": 20.0, "cpu_time_us": 20.0,
            },
        ]
        summary = self.profile._summarize(rows, limit=10, threads=16)
        transitions = summary["by_transition"]
        self.assertEqual(transitions[0]["from_layer"], 3)
        self.assertEqual(transitions[0]["to_layer"], 4)
        self.assertEqual(transitions[0]["gap_us"], 60.0)

    def test_summary_selects_representative_layer_topologies(self) -> None:
        rows = [
            {"mode": "prefill", "kernel": "deepseek_mla_attention_f32",
             "op": "mla_attention", "layer": 0, "time_us": 100.0,
             "cpu_time_us": 1200.0},
            {"mode": "prefill", "kernel": "moe_swiglu_expert_forward_bf16",
             "op": "moe_experts", "layer": 0, "time_us": 50.0,
             "cpu_time_us": 600.0},
            {"mode": "prefill", "kernel": "flash_attention",
             "op": "attention", "layer": 1, "time_us": 80.0,
             "cpu_time_us": 1000.0},
            {"mode": "prefill", "kernel": "gemm_q4",
             "op": "mlp_down", "layer": 1, "time_us": 40.0,
             "cpu_time_us": 500.0},
        ]
        summary = self.profile._summarize(rows, limit=10, threads=16)
        representatives = {
            row["topology"]: row["layer"] for row in summary["representative_layers"]
        }
        self.assertEqual(representatives, {
            "mla_attention+moe": 0,
            "full_attention+dense_mlp": 1,
        })

    def test_legacy_csv_without_cpu_time_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "mode,kernel,op,layer,time_us,token_id\n"
                "prefill,kernel,op,0,25.0,0\n",
                encoding="utf-8",
            )
            rows = self.profile._load_profile_rows(path)

        self.assertEqual(rows[0]["cpu_time_us"], 0.0)

    def test_prompt_token_file_is_truncated_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.json"
            path.write_text(json.dumps({"token_ids": [7, 11, 13, 17]}), encoding="utf-8")
            token_ids = self.profile._load_prompt_token_ids(path, limit=3)

        self.assertEqual(token_ids, [7, 11, 13])
        self.assertEqual(
            self.profile._token_ids_sha256(token_ids),
            "0d418e4514df520288cb2176052bf99455a4b7953c7baac90117f924b7eb7b3e",
        )

    def test_prompt_token_file_rejects_short_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.txt"
            path.write_text("7, 11\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2 IDs, but 3 were requested"):
                self.profile._load_prompt_token_ids(path, limit=3)


if __name__ == "__main__":
    unittest.main()
