#!/usr/bin/env python3
"""Run all trajectory comparisons, retaining numerical FAIL rather than stopping early.

Uses pinned independent Clef topology and PyTorch autograd. Exit 0: parity PASS;
1: numerical FAIL; 2: missing evidence. This is not a second training workflow.
"""

import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_v8_clef_full_training import FullHeadTraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="RNE BF16 projection operands, FP32 masters and other operations",
    )
    parser.add_argument(
        "--released",
        action="store_true",
        help="Released head geometry/weights instead of the complete small topology",
    )
    parser.add_argument(
        "--same-state",
        action="store_true",
        help="Diagnostic only: copy native weights and AdamW state into PyTorch before each step",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    case = unittest.TestCase()
    try:
        report = FullHeadTraining.certify(
            case,
            released=args.released,
            mixed=args.mixed,
            same_state=args.same_state,
            diagnostic=True,
            artifacts=args.report.with_suffix(""),
        )
    except unittest.SkipTest as exc:
        report = dict(numerical_parity="NOT_TESTED", reason=str(exc))
    finally:
        case.doCleanups()
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"PASS": 0, "FAIL": 1, "NOT_TESTED": 2}[report["numerical_parity"]]


if __name__ == "__main__":
    raise SystemExit(main())
