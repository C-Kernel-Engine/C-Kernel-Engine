from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "version/v8/scripts/run_zip_qwen35_moe_worker_v8.py"
RANK_PATH = ROOT / "version/v8/scripts/run_zip_qwen35_prefill_rank_v8.py"
PRELOAD_PATH = ROOT / "benchmarks/zip/ck_zip_moe_preload.c"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_zip_protocol_header_has_fixed_cross_process_size() -> None:
    worker = _load_module("ck_zip_qwen35_worker_test", WORKER_PATH)
    assert worker.HEADER.size == 80
    packed = worker.HEADER.pack(
        worker.MAGIC,
        worker.VERSION,
        7,
        512,
        2048,
        512,
        256,
        8,
        399,
        113,
        worker.REQUEST,
        0,
        925696,
        3616,
        3616,
        925696,
    )
    assert len(packed) == 80
    assert worker.HEADER.unpack(packed)[0:3] == (worker.MAGIC, worker.VERSION, 7)


def test_zip_rank_default_tokens_are_deterministic() -> None:
    rank = _load_module("ck_zip_qwen35_rank_test", RANK_PATH)
    args = type("Args", (), {"tokens_json": None, "token_count": 512})()
    tokens = rank._tokens(args)
    assert len(tokens) == 512
    digest = hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    assert digest == "aadaa92bbcd4498e77cceaa566ab890ac29fa5a8154068e61c6972710cea8408"


def test_zip_preload_compiles_strictly(tmp_path: Path) -> None:
    output = tmp_path / "libck_zip_moe_preload.so"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(output),
            str(PRELOAD_PATH),
            "-ldl",
            "-pthread",
        ],
        cwd=ROOT,
        check=True,
    )
    assert output.is_file() and output.stat().st_size > 0


def test_zip_transport_does_not_enter_numerical_kernel_sources() -> None:
    kernel_source = (ROOT / "src/kernels/axpy_kernels.c").read_text(encoding="utf-8")
    transport_source = PRELOAD_PATH.read_text(encoding="utf-8")
    assert "CK_ZIP_RESEARCH" not in kernel_source
    assert "socket(" not in kernel_source
    assert "CK_ZIP_RESEARCH_ROLE" in transport_source
    assert "moe_swiglu_expert_forward_q4k_q5k_parallel_workspace" in transport_source
    assert "moe_swiglu_shared_forward_q8_0_gated_workspace" in transport_source
