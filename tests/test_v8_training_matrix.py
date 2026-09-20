from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_training_matrix_v8.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_training_matrix_v8_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V8TrainingMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = _load()

    def _args(self, root: Path, workflow: Path, profiles: list[str]):
        args = self.matrix._parser().parse_args([
            "--run-root", str(root / "runs"), "--json-out", str(root / "matrix.json"),
            "--workflow", str(workflow), "--python", sys.executable,
            "--epochs", "2", "--grad-accum", "2", "--max-train-tokens", "8",
            "--max-validation-tokens", "4", "--case-timeout", "0.15",
            *sum((["--profile", profile] for profile in profiles), []),
        ])
        args.run_root = args.run_root.resolve(); args.report = args.report.resolve()
        args.workflow = args.workflow.resolve(); args.corpus = args.corpus.resolve()
        return args

    def test_malformed_stale_and_timeout_cases_continue_and_publish(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fake = root / "fake.py"
            fake.write_text(
                """import json, pathlib, sys, time\na=sys.argv\nout=pathlib.Path(a[a.index('--json-out')+1]); case=a[a.index('--matrix-case-id')+1]\nout.parent.mkdir(parents=True, exist_ok=True)\nif case == '4l_dense': out.write_text('{bad')\nelif case == '6l_gqa': out.write_text(json.dumps({'schema':'cke.v8.training_workflow.v1','status':'PASS','passed':True,'matrix_identity':{'run_id':'old','case_id':case,'profile':case}}))\nelse: time.sleep(2)\n""",
                encoding="utf-8",
            )
            args = self._args(root, fake, ["4l_dense", "6l_gqa", "10l_gqa"])
            result = self.matrix.run(args)
            self.assertFalse(result["passed"])
            self.assertEqual([row["failure_kind"] for row in result["cases"]],
                             ["malformed_report", "stale_report", "timeout"])
            published = json.loads(args.report.read_text(encoding="utf-8"))
            self.assertEqual(len(published["cases"]), 3)
            self.assertEqual(published["status"], "FAIL")

    def test_launch_errors_continue_and_replace_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fake = root / "unused.py"
            args = self._args(root, fake, ["4l_dense", "6l_gqa"])
            args.python = str(root / "missing-python")
            stale = args.run_root / "4l_dense" / "training_workflow.json"
            stale.parent.mkdir(parents=True); stale.write_text('{"passed":true}\n', encoding="utf-8")
            result = self.matrix.run(args)
            self.assertEqual([row["failure_kind"] for row in result["cases"]],
                             ["launch_error", "launch_error"])
            self.assertFalse(stale.exists())
            self.assertEqual(len(json.loads(args.report.read_text())["cases"]), 2)

    def test_complete_configuration_and_data_identity_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run_dir = root / "case"; dataset = run_dir / "dataset"; dataset.mkdir(parents=True)
            splits = {}
            for name, count in (("train", 8), ("validation", 4)):
                path = dataset / f"{name}_token_ids.i32"; path.write_bytes(bytes(count * 4))
                splits[name] = {"tokens": count, "token_ids": str(path), "token_ids_sha256": self.matrix._sha256(path)}
            expected = {"architecture":"qwen3_style_dense_reduced","dtype":"fp32","layers":6,"d_model":32,
                        "hidden":64,"heads":4,"kv_heads":2,"seq_len":32,"vocab_size":384,
                        "tokenizer":"bpe","epochs":2,"grad_accum":2,"optimizer":"generated_c_adamw",
                        "lr":3e-4,"beta1":.9,"beta2":.999,"eps":1e-8,"weight_decay":.01}
            report_path = run_dir / "training_workflow.json"; report_path.write_text("{}")
            document = {"schema":"cke.v8.training_workflow.v1","status":"PASS","passed":True,
                        "configuration":dict(expected),"matrix_identity":{"run_id":"run","case_id":"6l_gqa","profile":"6l_gqa",
                        "run_dir":str(run_dir),"report":str(report_path)},
                        "execution":{"git_commit":"abc"},"corpus":{"spec_sha256":"def","splits":splits,
                        "tokenizer":{"mode":"bpe","name":"cke_true_bpe_v1","vocab_size":384,"training_split_only":True}}}
            kwargs = dict(expected=expected, run_dir=run_dir, report_path=report_path, matrix_run_id="run",
                          case_id="6l_gqa", git_commit="abc", corpus_sha256="def",
                          max_train_tokens=8, max_validation_tokens=4)
            self.assertEqual(self.matrix._validate_case_report(document, **kwargs), [])
            for key in expected:
                altered = json.loads(json.dumps(document)); altered["configuration"][key] = "wrong"
                self.assertIn(f"configuration.{key}", self.matrix._validate_case_report(altered, **kwargs))


if __name__ == "__main__":
    unittest.main()
