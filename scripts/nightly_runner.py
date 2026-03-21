#!/usr/bin/env python3
"""
C-Kernel-Engine Nightly Test Runner

Runs all tests without stopping on failure, captures results,
tracks performance regressions, and generates summary reports.

Usage:
    python3 scripts/nightly_runner.py                    # Run all tests
    python3 scripts/nightly_runner.py --quick            # Quick subset only
    python3 scripts/nightly_runner.py --json report.json # Output JSON report
    python3 scripts/nightly_runner.py --save-baseline    # Save current perf as baseline
    python3 scripts/nightly_runner.py --category quant   # Run specific category

Categories:
    - kernels:  Core kernel tests (gemm, relu, softmax, etc.)
    - bf16:     BF16 precision tests
    - quant:    Quantization tests (Q4_K, Q6_K, etc.)
    - training: Training/backward pass tests
    - parity:   PyTorch parity tests
    - bench:    Benchmarks
    - all:      Everything (default)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
UNITTEST_DIR = ROOT / "unittest"
BF16_DIR = UNITTEST_DIR / "bf16"
BASELINE_FILE = ROOT / ".test_baseline.json"

# Keep pass logs compact, but preserve substantially more context for failures.
PASS_STDOUT_CHARS = 5000
PASS_STDERR_CHARS = 2000
FAIL_STDOUT_CHARS = 40000
FAIL_STDERR_CHARS = 12000


def _trim_output(text: str, limit: int, keep_head_tail: bool = False) -> str:
    if not text or len(text) <= limit:
        return text
    if not keep_head_tail:
        return text[-limit:]
    marker = "\n... [truncated] ...\n"
    head = max(0, int(limit * 0.4))
    tail = max(0, limit - head - len(marker))
    return text[:head] + marker + text[-tail:]


def parse_sub_tests(stdout: str) -> list:
    """
    Parse test output to extract individual sub-test results.

    Looks for patterns like:
      test_name  max_diff=1.23e-05  tol=1e-05  [PASS]
      kernel_name  max_diff=0.00e+00  tol=1e-06  [PASS]

    And performance lines like:
      kernel_name  123.4            45.6             2.71x
    """
    sub_tests = []

    # Pattern for accuracy results: name  max_diff=X  tol=Y  [PASS/FAIL]
    # Handles ANSI color codes like [92mPASS[0m
    accuracy_pattern = re.compile(
        r'^\s*(\S+(?:\s+\([^)]+\))?)\s+'  # test name (may include parentheses)
        r'max_diff=(\d+\.?\d*e?[+-]?\d*)\s+'  # max_diff value
        r'tol=(\d+\.?\d*e?[+-]?\d*)\s+'  # tolerance value
        r'\[(?:\x1b\[\d+m)?(PASS|FAIL)(?:\x1b\[\d+m)?\]',  # status with optional ANSI
        re.MULTILINE
    )

    # Pattern for tabular accuracy results (comprehensive GEMV tests):
    # name     MxK     max_diff    mean_diff    tol    [PASS/FAIL]
    # Example: Q4_K_tiny     1x256     1.14e-05     1.14e-05      1e-03  [PASS]
    tabular_accuracy_pattern = re.compile(
        r'^\s*(\S+)\s+'  # test name
        r'(\d+x\d+)\s+'  # dimensions MxK
        r'(\d+\.?\d*e?[+-]?\d*)\s+'  # max_diff
        r'(\d+\.?\d*e?[+-]?\d*)\s+'  # mean_diff
        r'(\d+\.?\d*e?[+-]?\d*)\s+'  # tolerance
        r'\[?(?:\x1b\[\d+m)?(PASS|FAIL)(?:\x1b\[\d+m)?\]?',  # status
        re.MULTILINE
    )

    # Pattern for "Test N: name" followed by "[PASS] max_diff = X" format
    # Example:
    #   Test 1: Q4_K Dequantization
    #     [PASS] max_diff = 0.00e+00
    test_n_pattern = re.compile(
        r'Test\s+\d+:\s+(.+?)[\n\r]+\s*'  # "Test N: name" line
        r'\[(?:\x1b\[\d+m)?(PASS|FAIL)(?:\x1b\[\d+m)?\]\s+'  # [PASS/FAIL]
        r'max_diff\s*=\s*(\d+\.?\d*e?[+-]?\d*)',  # max_diff = X
        re.MULTILINE
    )

    # Pattern for performance results: name  pytorch_time  c_time  speedup
    perf_pattern = re.compile(
        r'^\s*(\S+(?:\s+\([^)]+\))?)\s+'  # test name
        r'(\d+\.?\d*)\s+'  # pytorch time (us)
        r'(\d+\.?\d*)\s+'  # c kernel time (us)
        r'(?:\x1b\[\d+m)?(\d+\.?\d*)x',  # speedup with optional ANSI
        re.MULTILINE
    )

    # Extract accuracy results from standard format
    accuracy_results = {}
    for match in accuracy_pattern.finditer(stdout):
        name = match.group(1).strip()
        max_diff = float(match.group(2))
        tolerance = float(match.group(3))
        status = match.group(4).lower()
        accuracy_results[name] = {
            'max_diff': max_diff,
            'tolerance': tolerance,
            'status': status
        }

    # Also extract from tabular format (comprehensive GEMV tests)
    for match in tabular_accuracy_pattern.finditer(stdout):
        name = match.group(1).strip()
        # dimensions = match.group(2)  # MxK, not stored for now
        max_diff = float(match.group(3))
        # mean_diff = float(match.group(4))  # not stored for now
        tolerance = float(match.group(5))
        status = match.group(6).lower()
        # Don't overwrite if already found
        if name not in accuracy_results:
            accuracy_results[name] = {
                'max_diff': max_diff,
                'tolerance': tolerance,
                'status': status
            }

    # Extract from "Test N: name" format (Q4_K/Q6_K kernel tests)
    for match in test_n_pattern.finditer(stdout):
        name = match.group(1).strip()
        status = match.group(2).lower()
        max_diff = float(match.group(3))
        if name not in accuracy_results:
            accuracy_results[name] = {
                'max_diff': max_diff,
                'tolerance': None,  # Not provided in this format
                'status': status
            }

    # Extract performance results
    perf_results = {}
    for match in perf_pattern.finditer(stdout):
        name = match.group(1).strip()
        pytorch_time = float(match.group(2))
        c_time = float(match.group(3))
        speedup = float(match.group(4))
        perf_results[name] = {
            'pytorch_time_us': pytorch_time,
            'c_time_us': c_time,
            'speedup': speedup
        }

    # Merge accuracy and performance results
    all_names = set(accuracy_results.keys()) | set(perf_results.keys())
    for name in all_names:
        acc = accuracy_results.get(name, {})
        perf = perf_results.get(name, {})

        sub_test = SubTestResult(
            name=name,
            status=acc.get('status', 'pass'),
            max_diff=acc.get('max_diff'),
            tolerance=acc.get('tolerance'),
            c_time_us=perf.get('c_time_us'),
            pytorch_time_us=perf.get('pytorch_time_us'),
            speedup=perf.get('speedup')
        )
        sub_tests.append(sub_test)

    return sub_tests


@dataclass
class SubTestResult:
    """Individual test within a test file (e.g., per-kernel result)."""
    name: str
    status: str  # "pass", "fail"
    max_diff: Optional[float] = None
    tolerance: Optional[float] = None
    c_time_us: Optional[float] = None
    pytorch_time_us: Optional[float] = None
    speedup: Optional[float] = None


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # "pass", "fail", "skip", "timeout"
    duration_sec: float
    stdout: str = ""
    stderr: str = ""
    error_msg: str = ""
    perf_metric: Optional[float] = None  # GFLOPS, samples/sec, etc.
    perf_unit: str = ""
    baseline_perf: Optional[float] = None
    perf_delta_pct: Optional[float] = None
    sub_tests: list = field(default_factory=list)  # List of SubTestResult


@dataclass
class TestSuite:
    name: str
    category: str
    test_file: Path
    timeout_sec: int = 120
    perf_pattern: Optional[str] = None  # Regex to extract perf metric
    ci_skip: bool = False  # Skip in CI mode (tests requiring full shared library)


# Tests to skip in CI mode (use --ci flag)
# Currently empty - SmolLM-135M is downloaded in CI workflow
CI_SKIP_TESTS = set()

# Define all test suites
TEST_SUITES = {
    # Core kernel tests (FP32)
    "gemm": TestSuite("GEMM", "kernels", UNITTEST_DIR / "test_gemm.py"),
    "gemm_variants": TestSuite("GEMM Variants", "kernels", UNITTEST_DIR / "test_gemm_variants.py"),
    "gemm_microkernel": TestSuite("GEMM Microkernel", "kernels", UNITTEST_DIR / "test_gemm_microkernel.py"),
    "gemm_fused": TestSuite("GEMM Fused", "kernels", UNITTEST_DIR / "test_gemm_fused.py"),
    "relu": TestSuite("ReLU", "kernels", UNITTEST_DIR / "test_relu.py"),
    "gelu": TestSuite("GELU", "kernels", UNITTEST_DIR / "test_gelu.py"),
    "sigmoid": TestSuite("Sigmoid", "kernels", UNITTEST_DIR / "test_sigmoid.py"),
    "softmax": TestSuite("Softmax", "kernels", UNITTEST_DIR / "test_softmax.py"),
    "swiglu": TestSuite("SwiGLU", "kernels", UNITTEST_DIR / "test_swiglu.py"),
    "layernorm": TestSuite("LayerNorm", "kernels", UNITTEST_DIR / "test_layernorm.py"),
    "rmsnorm": TestSuite("RMSNorm", "kernels", UNITTEST_DIR / "test_rmsnorm.py"),
    "rope": TestSuite("RoPE", "kernels", UNITTEST_DIR / "test_rope.py"),
    "embedding": TestSuite("Embedding", "kernels", UNITTEST_DIR / "test_embedding.py"),
    "attention": TestSuite("Attention", "kernels", UNITTEST_DIR / "test_attention.py"),
    "attention_sliding": TestSuite("Attention Sliding Window", "kernels", UNITTEST_DIR / "test_attention_sliding_contract.py"),
    "kv_cache": TestSuite("KV Cache Attention", "kernels", UNITTEST_DIR / "test_kv_cache_attention.py"),
    "kv_cache_decode": TestSuite("KV Cache Decode", "kernels", UNITTEST_DIR / "test_kv_cache_layer_decode.py"),
    "fused_attention_decode": TestSuite("Fused Attention Decode", "kernels", UNITTEST_DIR / "test_fused_attention_decode.py"),
    "fused_swiglu_decode": TestSuite("Fused SwiGLU Decode", "kernels", UNITTEST_DIR / "test_fused_swiglu_decode.py"),
    "mlp": TestSuite("MLP", "kernels", UNITTEST_DIR / "test_mlp.py"),
    "cross_entropy": TestSuite("Cross Entropy", "kernels", UNITTEST_DIR / "test_cross_entropy.py"),
    "optimizer": TestSuite("Optimizer (AdamW/SGD)", "training", UNITTEST_DIR / "test_optimizer.py"),
    "lm_head_litmus": TestSuite("LM Head Litmus", "kernels", UNITTEST_DIR / "test_lm_head_litmus.py"),
    "vision": TestSuite("Vision", "kernels", UNITTEST_DIR / "test_vision.py"),
    # NOTE: Orchestration test disabled - v6.5 uses generated code with local helpers,
    # not orchestration layer. Use llamacpp-parity-full for quantized kernel validation.
    # "orchestration": TestSuite("Orchestration", "kernels", UNITTEST_DIR / "test_orchestration_layer.py"),

    # BF16 tests
    "relu_bf16": TestSuite("ReLU BF16", "bf16", BF16_DIR / "test_relu_bf16.py"),
    "gelu_bf16": TestSuite("GELU BF16", "bf16", BF16_DIR / "test_gelu_bf16.py"),
    "sigmoid_bf16": TestSuite("Sigmoid BF16", "bf16", BF16_DIR / "test_sigmoid_bf16.py"),
    "swiglu_bf16": TestSuite("SwiGLU BF16", "bf16", BF16_DIR / "test_swiglu_bf16.py"),
    "layernorm_bf16": TestSuite("LayerNorm BF16", "bf16", BF16_DIR / "test_layernorm_bf16.py"),
    "rmsnorm_bf16": TestSuite("RMSNorm BF16", "bf16", BF16_DIR / "test_rmsnorm_bf16.py"),
    "rope_bf16": TestSuite("RoPE BF16", "bf16", BF16_DIR / "test_rope_bf16.py"),
    "embedding_bf16": TestSuite("Embedding BF16", "bf16", BF16_DIR / "test_embedding_bf16.py"),
    "attention_bf16": TestSuite("Attention BF16", "bf16", BF16_DIR / "test_attention_bf16.py"),
    "mlp_bf16": TestSuite("MLP BF16", "bf16", BF16_DIR / "test_mlp_bf16.py"),
    "cross_entropy_bf16": TestSuite("Cross Entropy BF16", "bf16", BF16_DIR / "test_cross_entropy_bf16.py"),

    # Quantization tests
    # NOTE: test_quant_kernels.py and test_q4_k_quantize.py removed - use llamacpp-parity-full
    # for authoritative kernel correctness testing against llama.cpp reference
    "q4k_kernels": TestSuite("Q4_K Kernels", "quant", UNITTEST_DIR / "test_q4k_kernels.py"),
    "q6k_kernels": TestSuite("Q6_K Kernels", "quant", UNITTEST_DIR / "test_q6k_kernels.py"),
    "q4_k_q8_k_matvec": TestSuite("Q4_K x Q8_K MatVec", "quant", UNITTEST_DIR / "test_q4_k_q8_k_matvec.py"),

    # Training/backward tests
    "softmax_backward": TestSuite("Softmax Backward", "training", UNITTEST_DIR / "test_softmax_backward.py"),
    "attention_backward": TestSuite("Attention Backward", "training", UNITTEST_DIR / "test_attention_backward.py"),
    "deltanet_backward": TestSuite("DeltaNet Backward", "training", ROOT / "tests" / "test_deltanet.py"),

    # Parity tests
    "pytorch_parity": TestSuite("PyTorch Parity", "parity", UNITTEST_DIR / "test_pytorch_parity.py", timeout_sec=300),
    "rope_pairwise_layout": TestSuite(
        "RoPE Pairwise Layout (Llama)",
        "parity",
        ROOT / "version" / "v7" / "scripts" / "parity" / "test_check_decode_attention_contract.py",
        timeout_sec=120,
    ),
}

# Make targets to run (non-Python tests)
MAKE_TARGETS = {
    "litmus": {
        "name": "Litmus Test",
        "category": "parity",
        "target": "litmus",
        "timeout_sec": 180,
    },
    # NOTE: Layer parity disabled - v6.5 uses generated code, not orchestration layer.
    # Use llamacpp-parity-full for quantized kernel validation.
    # "layer_parity": {
    #     "name": "Layer Parity",
    #     "category": "parity",
    #     "target": "layer-parity",
    #     "timeout_sec": 300,
    # },
    # TODO: Re-enable in v7 when training pipeline is updated
    # "smollm_parity": {
    #     "name": "SmolLM Train Parity",
    #     "category": "parity",
    #     "target": "smollm-train-parity",
    #     "timeout_sec": 600,
    # },
    "llamacpp_parity": {
        "name": "llama.cpp Parity (Full)",
        "category": "parity",
        "target": "llamacpp-parity-full",
        "timeout_sec": 900,  # Full test takes longer
    },
    "flash_attention": {
        "name": "Flash Attention (50K+)",
        "category": "kernels",
        "target": "test-flash-attention",
        "timeout_sec": 1800,
    },
    "threadpool_parity": {
        "name": "Thread Pool GEMV Parity (serial vs dispatch)",
        "category": "parity",
        "target": "test-threadpool-parity",
        "timeout_sec": 300,
    },
    "v6_6_contracts": {
        "name": "v6.6 Tooling Contracts",
        "category": "parity",
        "target": "v6.6-validate-contracts",
        "timeout_sec": 180,
    },
    "v6_6_kernel_map_gate": {
        "name": "v6.6 Kernel Map Gate",
        "category": "parity",
        "target": "v6.6-kernel-map-gate",
        "timeout_sec": 240,
    },
    "v6_6_model_matrix": {
        "name": "v6.6 Model Matrix (Build)",
        "category": "parity",
        "target": "v6.6-validate-matrix-nightly",
        "timeout_sec": 2400,
    },
    "v7_backprop_long_epoch_nightly": {
        "name": "v7 Backprop Long-Epoch Drift",
        "category": "training",
        "target": "v7-backprop-long-epoch-nightly",
        "timeout_sec": 2400,
    },
    "v7_kernel_parity_train": {
        "name": "v7 Backprop Kernel Parity",
        "category": "training",
        "target": "v7-kernel-parity-train",
        "timeout_sec": 1800,
    },
    "v7_ir_visualizer_e2e": {
        "name": "v7 IR Visualizer E2E (full)",
        "category": "parity",
        "target": "v7-ir-visualizer-e2e-nightly",
        "timeout_sec": 5400,
    },
    "v7_visualizer_health": {
        "name": "v7 Visualizer Health (L1+L2)",
        "category": "parity",
        "target": "v7-visualizer-health",
        "timeout_sec": 60,
    },
    "v7_visualizer_generated_e2e": {
        "name": "v7 Visualizer Generated E2E (L3)",
        "category": "parity",
        "target": "v7-visualizer-generated-e2e",
        "timeout_sec": 600,
    },
    "v7_core_stabilization_nightly": {
        "name": "v7 Core Stabilization Matrix",
        "category": "training",
        "target": "v7-stabilization-nightly",
        "timeout_sec": 7200,
    },
}

# Benchmark targets with perf extraction
BENCH_TARGETS = {
    "bench_gemm": {
        "name": "GEMM Benchmark",
        "category": "bench",
        "target": "bench_gemm",
        "timeout_sec": 300,
        "perf_pattern": r"(\d+\.?\d*)\s*GFLOPS",
        "perf_unit": "GFLOPS",
    },
}

# Quick subset for fast validation
QUICK_TESTS = [
    "gemm", "relu", "softmax", "rmsnorm", "attention", "attention_sliding",
    "deltanet_backward",
    "relu_bf16", "rmsnorm_bf16",
    "q4k_kernels",
]

MAKE_TARGET_FAILURE_ARTIFACTS = {
    "v6.6-validate-matrix-nightly": ROOT / "version" / "v6.6" / "tools" / "model_matrix_report_latest.json",
    "v7-ir-visualizer-e2e-nightly": ROOT / "version" / "v7" / ".cache" / "reports" / "ir_visualizer_e2e_latest.json",
    "v7-visualizer-health": ROOT / "version" / "v7" / ".cache" / "reports" / "visualizer_health_latest.json",
    "v7-visualizer-generated-e2e": ROOT / "version" / "v7" / ".cache" / "reports" / "visualizer_generated_e2e_latest.json",
    "v7-stabilization-nightly": ROOT / "version" / "v7" / ".cache" / "reports" / "training_stabilization_scorecard_latest.json",
}


def _load_json_if_fresh(path: Path, *, start_ts: float) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        # Ignore stale artifacts from a previous run.
        if path.stat().st_mtime < (start_ts - 5.0):
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _summarize_make_failure_artifact(target: str, *, start_ts: float) -> str:
    path = MAKE_TARGET_FAILURE_ARTIFACTS.get(target)
    if path is None:
        return ""
    payload = _load_json_if_fresh(path, start_ts=start_ts)
    if payload is None:
        return ""

    prefix = f"artifact={path}"
    if target == "v6.6-validate-matrix-nightly":
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return prefix
        failing = [r for r in rows if isinstance(r, dict) and r.get("overall") != "PASS"]
        if not failing:
            summary = payload.get("summary")
            return f"{prefix}; summary={summary}"
        details = []
        for row in failing[:3]:
            details.append(
                f"{row.get('model')}:{row.get('overall')}:{row.get('note')}"
            )
        if len(failing) > 3:
            details.append(f"+{len(failing) - 3} more")
        return f"{prefix}; failing_rows={' | '.join(details)}"

    if target == "v7-ir-visualizer-e2e-nightly":
        checks = payload.get("checks")
        if not isinstance(checks, list):
            return prefix
        failing = [c for c in checks if isinstance(c, dict) and not c.get("passed", False)]
        if not failing:
            ok = payload.get("ok")
            return f"{prefix}; ok={ok}"
        details = []
        for chk in failing[:4]:
            details.append(f"{chk.get('name')}:{chk.get('detail')}")
        if len(failing) > 4:
            details.append(f"+{len(failing) - 4} more")
        return f"{prefix}; failing_checks={' | '.join(details)}"

    if target in ("v7-visualizer-health", "v7-visualizer-generated-e2e"):
        if payload.get("skipped"):
            return f"{prefix}; skipped={payload.get('reason', 'no runs')}"
        total = payload.get("total_checks", payload.get("total", 0))
        passed = payload.get("passed", 0)
        failed = payload.get("failed", 0)
        warnings = payload.get("warnings", 0)
        if failed == 0:
            return f"{prefix}; {passed}/{total} passed ({warnings} warnings)"
        # Show failing stages/suites
        stages = payload.get("stages", payload.get("suites", []))
        failing_names = [
            s.get("stage", s.get("suite", "?"))
            for s in stages if isinstance(s, dict) and s.get("failed", 0) > 0
        ]
        return f"{prefix}; {failed}/{total} failed: {', '.join(failing_names[:5])}"

    if target == "v7-stabilization-nightly":
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        tokenizer = payload.get("tokenizer_gates") if isinstance(payload.get("tokenizer_gates"), list) else []
        matrix = payload.get("matrix_cases") if isinstance(payload.get("matrix_cases"), list) else []
        failing_tokenizer = [g for g in tokenizer if isinstance(g, dict) and g.get("status") != "PASS"]
        failing_matrix = [c for c in matrix if isinstance(c, dict) and c.get("status") != "PASS"]
        parts = [
            f"summary=passed:{summary.get('passed')} tokenizer:{summary.get('passed_tokenizer_gates')}/{summary.get('total_tokenizer_gates')} matrix:{summary.get('passed_matrix_cases')}/{summary.get('total_matrix_cases')}"
        ]
        if failing_tokenizer:
            parts.append(
                "tokenizer_fail="
                + " | ".join(f"{g.get('mode')}:{g.get('metrics', {})}" for g in failing_tokenizer[:2])
            )
        if failing_matrix:
            details = []
            for case in failing_matrix[:3]:
                metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
                details.append(
                    f"{case.get('case_id')}:{metrics.get('failed_stage_ids')}"
                )
            if len(failing_matrix) > 3:
                details.append(f"+{len(failing_matrix) - 3} more")
            parts.append("matrix_fail=" + " | ".join(details))
        return f"{prefix}; {'; '.join(parts)}"

    return prefix


def run_python_test(suite: TestSuite, verbose: bool = False) -> TestResult:
    """Run a Python test file and capture results."""
    if not suite.test_file.exists():
        return TestResult(
            name=suite.name,
            category=suite.category,
            status="skip",
            duration_sec=0.0,
            error_msg=f"Test file not found: {suite.test_file}",
        )

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(suite.test_file)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=suite.timeout_sec,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        duration = time.time() - start

        # Extract perf metric if pattern provided
        perf_metric = None
        if suite.perf_pattern:
            match = re.search(suite.perf_pattern, result.stdout)
            if match:
                perf_metric = float(match.group(1))

        # Parse sub-test results from output
        sub_tests = parse_sub_tests(result.stdout)

        if result.returncode == 0:
            return TestResult(
                name=suite.name,
                category=suite.category,
                status="pass",
                duration_sec=duration,
                stdout=_trim_output(result.stdout, PASS_STDOUT_CHARS),
                stderr=_trim_output(result.stderr, PASS_STDERR_CHARS),
                perf_metric=perf_metric,
                perf_unit=suite.perf_pattern and "unit" or "",
                sub_tests=sub_tests,
            )
        else:
            # Extract error message from output
            error_lines = []
            for line in (result.stdout + result.stderr).splitlines():
                if any(x in line.lower() for x in ["error", "fail", "assert", "exception"]):
                    error_lines.append(line)
            error_msg = "\n".join(error_lines[-10:]) if error_lines else "Non-zero exit code"

            return TestResult(
                name=suite.name,
                category=suite.category,
                status="fail",
                duration_sec=duration,
                stdout=_trim_output(result.stdout, FAIL_STDOUT_CHARS, keep_head_tail=True),
                stderr=_trim_output(result.stderr, FAIL_STDERR_CHARS, keep_head_tail=True),
                error_msg=error_msg,
                sub_tests=sub_tests,
            )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=suite.name,
            category=suite.category,
            status="timeout",
            duration_sec=suite.timeout_sec,
            error_msg=f"Test timed out after {suite.timeout_sec}s",
        )
    except Exception as e:
        return TestResult(
            name=suite.name,
            category=suite.category,
            status="fail",
            duration_sec=time.time() - start,
            error_msg=str(e),
        )


def run_make_target(target_info: dict, verbose: bool = False) -> TestResult:
    """Run a make target and capture results."""
    start = time.time()
    try:
        result = subprocess.run(
            ["make", target_info["target"]],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=target_info["timeout_sec"],
        )
        duration = time.time() - start

        # Extract perf metric if pattern provided
        perf_metric = None
        perf_unit = ""
        if "perf_pattern" in target_info:
            match = re.search(target_info["perf_pattern"], result.stdout)
            if match:
                perf_metric = float(match.group(1))
                perf_unit = target_info.get("perf_unit", "")

        status = "pass" if result.returncode == 0 else "fail"
        error_msg = ""
        if status == "fail":
            # Look for error indicators in both stdout and stderr
            # Match: [ERROR], [FAIL], error, Error, failed, FAILED
            combined = result.stdout + "\n" + result.stderr
            error_patterns = ["error", "[fail]", "[error]", "failed", "failure"]
            error_lines = []
            for line in combined.splitlines():
                line_lower = line.lower()
                if any(pat in line_lower for pat in error_patterns):
                    # Strip ANSI codes for cleaner output
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
                    if clean_line and clean_line not in error_lines:
                        error_lines.append(clean_line)
            error_msg = "\n".join(error_lines[-10:]) if error_lines else f"Exit code {result.returncode}"
            artifact_summary = _summarize_make_failure_artifact(target_info["target"], start_ts=start)
            if artifact_summary:
                error_msg = f"{error_msg}\n{artifact_summary}" if error_msg else artifact_summary

        sub_tests = parse_sub_tests(result.stdout)
        return TestResult(
            name=target_info["name"],
            category=target_info["category"],
            status=status,
            duration_sec=duration,
            stdout=_trim_output(
                result.stdout,
                FAIL_STDOUT_CHARS if status == "fail" else PASS_STDOUT_CHARS,
                keep_head_tail=(status == "fail"),
            ),
            stderr=_trim_output(
                result.stderr,
                FAIL_STDERR_CHARS if status == "fail" else PASS_STDERR_CHARS,
                keep_head_tail=(status == "fail"),
            ),
            error_msg=error_msg,
            perf_metric=perf_metric,
            perf_unit=perf_unit,
            sub_tests=sub_tests,
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=target_info["name"],
            category=target_info["category"],
            status="timeout",
            duration_sec=target_info["timeout_sec"],
            error_msg=f"Timed out after {target_info['timeout_sec']}s",
        )
    except Exception as e:
        return TestResult(
            name=target_info["name"],
            category=target_info["category"],
            status="fail",
            duration_sec=time.time() - start,
            error_msg=str(e),
        )


def load_baseline() -> dict:
    """Load performance baseline from file."""
    if BASELINE_FILE.exists():
        try:
            return json.loads(BASELINE_FILE.read_text())
        except:
            pass
    return {}


def save_baseline(results: list[TestResult]):
    """Save current perf metrics as baseline."""
    baseline = {}
    for r in results:
        if r.perf_metric is not None:
            baseline[r.name] = {
                "perf": r.perf_metric,
                "unit": r.perf_unit,
                "timestamp": datetime.now().isoformat(),
            }
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
    print(f"\nSaved baseline to {BASELINE_FILE}")


def compare_with_baseline(results: list[TestResult], baseline: dict):
    """Add baseline comparison to results."""
    for r in results:
        if r.name in baseline and r.perf_metric is not None:
            base_perf = baseline[r.name]["perf"]
            r.baseline_perf = base_perf
            if base_perf > 0:
                r.perf_delta_pct = ((r.perf_metric - base_perf) / base_perf) * 100


def print_summary(results: list[TestResult], start_time: datetime):
    """Print test run summary."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    timeout = sum(1 for r in results if r.status == "timeout")
    total_duration = sum(r.duration_sec for r in results)

    print("\n" + "=" * 70)
    print("  C-KERNEL-ENGINE NIGHTLY TEST REPORT")
    print("=" * 70)
    print(f"\n  Run started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time:   {total_duration:.1f}s")
    print()
    print(f"  PASSED:   {passed:>3}  {'█' * passed}")
    print(f"  FAILED:   {failed:>3}  {'█' * failed}" if failed else f"  FAILED:   {failed:>3}")
    print(f"  SKIPPED:  {skipped:>3}" if skipped else f"  SKIPPED:  {skipped:>3}")
    print(f"  TIMEOUT:  {timeout:>3}" if timeout else f"  TIMEOUT:  {timeout:>3}")
    print(f"  ─────────────")
    print(f"  TOTAL:    {total:>3}")
    print()

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"pass": 0, "fail": 0, "skip": 0, "timeout": 0}
        categories[r.category][r.status] += 1

    print("  BY CATEGORY:")
    print("  " + "-" * 50)
    for cat, counts in sorted(categories.items()):
        status_str = f"✓{counts['pass']}"
        if counts['fail']:
            status_str += f" ✗{counts['fail']}"
        if counts['skip']:
            status_str += f" ○{counts['skip']}"
        if counts['timeout']:
            status_str += f" ⏱{counts['timeout']}"
        print(f"  {cat:<15} {status_str}")

    # Print failures
    failures = [r for r in results if r.status in ("fail", "timeout")]
    if failures:
        print("\n" + "=" * 70)
        print("  FAILURES")
        print("=" * 70)
        for r in failures:
            status_icon = "✗" if r.status == "fail" else "⏱"
            print(f"\n  {status_icon} {r.name} [{r.category}]")
            print(f"    Duration: {r.duration_sec:.1f}s")
            if r.error_msg:
                print("    Error details:")
                for line in r.error_msg.splitlines()[:10]:
                    print(f"      {line[:100]}")
            # For make targets, show summary section from stdout if available
            if r.stdout and "SUMMARY" in r.stdout:
                lines = r.stdout.splitlines()
                for i, line in enumerate(lines):
                    if "SUMMARY" in line:
                        # Print summary section (up to 15 lines)
                        print("    Test summary:")
                        for j in range(i, min(i + 15, len(lines))):
                            clean = re.sub(r'\x1b\[[0-9;]*m', '', lines[j]).strip()
                            if clean:
                                print(f"      {clean[:100]}")

    # Print performance regressions
    regressions = [r for r in results if r.perf_delta_pct is not None and r.perf_delta_pct < -10]
    if regressions:
        print("\n" + "=" * 70)
        print("  PERFORMANCE REGRESSIONS (>10% slower)")
        print("=" * 70)
        for r in regressions:
            print(f"\n  ⚠ {r.name}")
            print(f"    Baseline: {r.baseline_perf:.1f} {r.perf_unit}")
            print(f"    Current:  {r.perf_metric:.1f} {r.perf_unit}")
            print(f"    Delta:    {r.perf_delta_pct:+.1f}%")

    print("\n" + "=" * 70)

    # Return exit code
    return 0 if failed == 0 and timeout == 0 else 1


def save_json_report(results: list[TestResult], filepath: Path, start_time: datetime):
    """Save results as JSON report."""
    # Count total sub-tests
    total_sub_tests = sum(len(r.sub_tests) for r in results)
    passed_sub_tests = sum(
        sum(1 for st in r.sub_tests if st.status == "pass")
        for r in results
    )
    failed_sub_tests = sum(
        sum(1 for st in r.sub_tests if st.status == "fail")
        for r in results
    )

    # Convert results to dicts, handling sub_tests properly
    results_dicts = []
    for r in results:
        d = asdict(r)
        # Convert SubTestResult objects to dicts
        d['sub_tests'] = [asdict(st) for st in r.sub_tests]
        # Don't include full stdout/stderr in JSON (too large)
        d['stdout'] = ""
        d['stderr'] = ""
        results_dicts.append(d)

    report = {
        "timestamp": start_time.isoformat(),
        "duration_sec": sum(r.duration_sec for r in results),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "pass"),
            "failed": sum(1 for r in results if r.status == "fail"),
            "skipped": sum(1 for r in results if r.status == "skip"),
            "timeout": sum(1 for r in results if r.status == "timeout"),
            "sub_tests_total": total_sub_tests,
            "sub_tests_passed": passed_sub_tests,
            "sub_tests_failed": failed_sub_tests,
        },
        "results": results_dicts,
    }
    filepath.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report saved to: {filepath}")


def update_nightly_index(results_dir=None):
    """Update the nightly results index.json file."""
    if results_dir:
        reports_dir = Path(results_dir)
    else:
        reports_dir = ROOT / "docs" / "site" / "nightly-results"
    index_file = reports_dir / "index.json"

    index_data = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "reports": []
    }

    # Get all report files, sorted newest first
    report_files = sorted(
        [f for f in os.listdir(reports_dir) if f.startswith("report-") and f.endswith(".json")],
        reverse=True
    )[:30]

    for filename in report_files:
        filepath = os.path.join(reports_dir, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            # Extract date from filename: report-YYYY-MM-DD.json
            date = filename.replace("report-", "").replace(".json", "")
            index_data["reports"].append({
                "date": date,
                "file": filename,
                "summary": data.get("summary", {}),
                "duration_sec": data.get("duration_sec", 0)
            })
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Created index.json with {len(index_data['reports'])} reports")


def main():
    parser = argparse.ArgumentParser(description="C-Kernel-Engine Nightly Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick subset only")
    parser.add_argument("--ci", action="store_true", help="CI mode: skip tests requiring full shared library")
    parser.add_argument("--category", type=str, help="Run specific category (kernels, bf16, quant, training, parity, bench)")
    parser.add_argument("--json", type=str, metavar="FILE", help="Save JSON report to file")
    parser.add_argument("--save-baseline", action="store_true", help="Save current perf as baseline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List all tests")
    parser.add_argument("--no-fail", action="store_true", help="Always return exit code 0 (for CI warning mode)")
    parser.add_argument("--update-index", action="store_true", help="Update nightly results index (for CI)")
    parser.add_argument("--results-dir", type=str, help="Directory for nightly results (used with --update-index)")
    args = parser.parse_args()

    # Handle --update-index (must be done before normal flow)
    if args.update_index:
        update_nightly_index(args.results_dir)
        return 0

    if args.list:
        print("\nAvailable tests:")
        print("-" * 50)
        for key, suite in sorted(TEST_SUITES.items()):
            exists = "✓" if suite.test_file.exists() else "✗"
            print(f"  {exists} {key:<25} [{suite.category}]")
        print("\nMake targets:")
        for key, info in sorted(MAKE_TARGETS.items()):
            print(f"    {key:<25} [{info['category']}]")
        print("\nBenchmarks:")
        for key, info in sorted(BENCH_TARGETS.items()):
            print(f"    {key:<25} [{info['category']}]")
        return 0

    # Determine which tests to run
    tests_to_run = []
    make_targets_to_run = []
    bench_targets_to_run = []

    if args.quick:
        tests_to_run = [k for k in QUICK_TESTS if k in TEST_SUITES]
    elif args.category:
        tests_to_run = [k for k, v in TEST_SUITES.items() if v.category == args.category]
        make_targets_to_run = [k for k, v in MAKE_TARGETS.items() if v["category"] == args.category]
        bench_targets_to_run = [k for k, v in BENCH_TARGETS.items() if v["category"] == args.category]
    else:
        tests_to_run = list(TEST_SUITES.keys())
        make_targets_to_run = list(MAKE_TARGETS.keys())
        bench_targets_to_run = list(BENCH_TARGETS.keys())

    # In CI mode, skip tests that require the full shared library
    if args.ci:
        skipped_count = len([t for t in tests_to_run if t in CI_SKIP_TESTS])
        tests_to_run = [t for t in tests_to_run if t not in CI_SKIP_TESTS]
        make_targets_to_run = [t for t in make_targets_to_run if t not in CI_SKIP_TESTS]
        if skipped_count > 0:
            print(f"  [CI MODE] Skipping {skipped_count} tests requiring full shared library")

    print("\n" + "=" * 70)
    print("  C-KERNEL-ENGINE NIGHTLY TEST RUNNER")
    print("=" * 70)
    print(f"\n  Tests to run: {len(tests_to_run)} Python tests")
    print(f"                {len(make_targets_to_run)} Make targets")
    print(f"                {len(bench_targets_to_run)} Benchmarks")
    print()

    start_time = datetime.now()
    results = []
    baseline = load_baseline()

    # Run Python tests
    for i, test_key in enumerate(tests_to_run, 1):
        suite = TEST_SUITES[test_key]
        print(f"  [{i}/{len(tests_to_run)}] {suite.name}...", end=" ", flush=True)
        result = run_python_test(suite, verbose=args.verbose)
        results.append(result)

        status_icon = {"pass": "✓", "fail": "✗", "skip": "○", "timeout": "⏱"}[result.status]
        print(f"{status_icon} ({result.duration_sec:.1f}s)")

        if args.verbose and result.status == "fail":
            print(f"      Error: {result.error_msg[:100]}")

    # Run make targets
    for i, target_key in enumerate(make_targets_to_run, 1):
        info = MAKE_TARGETS[target_key]
        print(f"  [make {i}/{len(make_targets_to_run)}] {info['name']}...", end=" ", flush=True)
        result = run_make_target(info, verbose=args.verbose)
        results.append(result)

        status_icon = {"pass": "✓", "fail": "✗", "skip": "○", "timeout": "⏱"}[result.status]
        print(f"{status_icon} ({result.duration_sec:.1f}s)")

    # Run benchmarks
    for i, bench_key in enumerate(bench_targets_to_run, 1):
        info = BENCH_TARGETS[bench_key]
        print(f"  [bench {i}/{len(bench_targets_to_run)}] {info['name']}...", end=" ", flush=True)
        result = run_make_target(info, verbose=args.verbose)
        results.append(result)

        status_icon = {"pass": "✓", "fail": "✗", "skip": "○", "timeout": "⏱"}[result.status]
        perf_str = f" {result.perf_metric:.1f} {result.perf_unit}" if result.perf_metric else ""
        print(f"{status_icon} ({result.duration_sec:.1f}s){perf_str}")

    # Compare with baseline
    compare_with_baseline(results, baseline)

    # Save baseline if requested
    if args.save_baseline:
        save_baseline(results)

    # Print summary
    exit_code = print_summary(results, start_time)

    # Save JSON if requested
    if args.json:
        save_json_report(results, Path(args.json), start_time)

    # --no-fail: always return 0 for CI warning mode
    if args.no_fail:
        return 0
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
