#!/usr/bin/env python3
"""
v7 deterministic 1-token parity harness.

Runs core C-kernel vs PyTorch parity checks at T=1:
- RMSNorm forward/backward parity
- SwiGLU forward/backward parity
- Cross-entropy (loss + gradient) parity

This is the correctness-first gate for v7.0.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[3]
UNITTEST_DIR = ROOT / "unittest"
QK_NORM_PARITY_SCRIPT = ROOT / "version" / "v7" / "scripts" / "check_qk_norm_backward_parity_v7.py"


def _import_parity_module():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    # Prefer conservative torch dispatch on mixed/older hosts.
    os.environ.setdefault("CK_TORCH_SAFE", "1")
    import test_pytorch_parity as parity  # noqa: E402
    return parity


def _report_to_dict(report: Any) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for r in getattr(report, "results", []):
        items.append(
            {
                "name": r.name,
                "passed": bool(r.passed),
                "max_diff": float(r.max_diff) if r.max_diff is not None else None,
                "tolerance": float(r.tolerance) if r.tolerance is not None else None,
            }
        )
    return {
        "test_name": getattr(report, "test_name", "unknown"),
        "passed": bool(report.all_passed()),
        "results": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v7 deterministic 1-token PyTorch parity checks.")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension for RMSNorm/SwiGLU parity.")
    parser.add_argument("--vocab", type=int, default=256, help="Vocab size for CE parity.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.d_model <= 0:
        print("ERROR: --d-model must be > 0", file=sys.stderr)
        return 2
    if args.vocab <= 1:
        print("ERROR: --vocab must be > 1", file=sys.stderr)
        return 2

    try:
        parity = _import_parity_module()
    except Exception as exc:
        print("ERROR: failed to import parity module from unittest/test_pytorch_parity.py", file=sys.stderr)
        print("DETAIL: %s" % exc, file=sys.stderr)
        print("HINT: install PyTorch in the active environment, e.g. `python -m pip install torch`", file=sys.stderr)
        return 2

    try:
        import numpy as np
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    except Exception:
        pass

    print("=" * 92)
    print("v7 1-TOKEN PARITY (deterministic fp32)")
    print("=" * 92)
    print("Config: T=1 D=%d V=%d seed=%d" % (args.d_model, args.vocab, args.seed))

    reports = []
    reports.append(parity.test_rmsnorm_parity(T=1, D=args.d_model, warmup=2, iterations=25))
    reports.append(parity.test_swiglu_parity(T=1, D=args.d_model, warmup=2, iterations=25))
    reports.append(parity.test_cross_entropy_parity(T=1, V=args.vocab, warmup=2, iterations=25))

    overall_pass = True
    report_payload = []
    for rep in reports:
        rep.print_report()
        rep_dict = _report_to_dict(rep)
        report_payload.append(rep_dict)
        overall_pass = overall_pass and rep_dict["passed"]

    passed = sum(1 for x in report_payload if x["passed"])
    total = len(report_payload)
    print("=" * 92)
    if overall_pass:
        print("SUMMARY: PASS (%d/%d test groups)" % (passed, total))
    else:
        print("SUMMARY: FAIL (%d/%d test groups passed)" % (passed, total))
    print("=" * 92)

    payload = {
        "mode": "v7-parity-1token",
        "config": {
            "T": 1,
            "D": args.d_model,
            "V": args.vocab,
            "seed": args.seed,
        },
        "overall_pass": overall_pass,
        "reports": report_payload,
    }

    # Add qk_norm backward parity as an explicit v7 training-kernel check.
    if QK_NORM_PARITY_SCRIPT.exists():
        qk_json: Optional[Path] = None
        if args.json_out is not None:
            qk_json = args.json_out.parent / "qk_norm_backward_parity_latest.json"
        cmd = [
            sys.executable,
            str(QK_NORM_PARITY_SCRIPT),
            "--num-tokens",
            "1",
        ]
        if qk_json is not None:
            cmd += ["--json-out", str(qk_json)]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        qk_payload: Dict[str, Any] = {
            "name": "qk_norm_backward_parity",
            "passed": proc.returncode == 0,
            "output": proc.stdout,
        }
        if qk_json is not None and qk_json.exists():
            try:
                qk_payload["details"] = json.loads(qk_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        payload["reports"].append(qk_payload)
        payload["overall_pass"] = bool(payload["overall_pass"] and qk_payload["passed"])
        overall_pass = payload["overall_pass"]

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON: %s" % args.json_out)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
