import importlib.util
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
            "mode,kernel,op,layer,time_us,cpu_time_us,token_id",
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


if __name__ == "__main__":
    unittest.main()
