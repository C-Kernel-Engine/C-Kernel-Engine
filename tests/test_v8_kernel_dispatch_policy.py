import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "version/v8/scripts/generate_kernel_dispatch_policy_v8.py"
OUTPUT_PATH = ROOT / "version/v8/src/ck_kernel_dispatch_policy_v8.inc"
Q4_MAP = ROOT / "version/v8/kernel_maps/gemm_nt_q4_k_q8_k.json"
Q6_MAP = ROOT / "version/v8/kernel_maps/gemm_nt_q6_k_q8_k.json"
PREFILL_SOURCE = ROOT / "version/v8/src/ck_parallel_prefill_v8.c"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_kernel_dispatch_policy_v8", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_dispatch_policy_is_fresh() -> None:
    generator = _load_generator()
    assert OUTPUT_PATH.read_text(encoding="utf-8") == generator.generate()


def test_q4_runtime_policy_owns_x16_shapes_and_tail_schedule() -> None:
    data = json.loads(Q4_MAP.read_text(encoding="utf-8"))
    policies = data["implementation"]["runtime_dispatch"]["policies"]
    prefill = policies["avx512_vnni_x16_prefill"]
    decode = policies["avx512_vnni_x16_decode_prepared"]

    assert prefill["status"] == "candidate"
    assert any(route.get("flags") == ["batched_tail"] for route in prefill["routes"])
    assert any(route.get("max_threads") == 20 for route in prefill["routes"])
    assert all(route.get("m") == 1 for route in decode["routes"])


def test_q6_runtime_policy_owns_shape_scheduling() -> None:
    data = json.loads(Q6_MAP.read_text(encoding="utf-8"))
    routes = data["implementation"]["runtime_dispatch"]["policies"]["prefill_schedule"]["routes"]

    assert routes[0] == {
        "min_m": 4,
        "max_m": 63,
        "n": 10240,
        "k": 5120,
        "tile_m": 16,
        "tile_n": 64,
        "flags": ["output_tiles", "compact_m4"],
    }
    assert routes[-1] == {"min_m": 1, "tile_m": 16, "tile_n": 256}


def test_shared_prefill_runtime_has_no_qwen36_shape_dispatch() -> None:
    source = PREFILL_SOURCE.read_text(encoding="utf-8").lower()
    assert "qwen36_recurrent_qkv" not in source
    assert "ck_should_use_qwen36" not in source
    assert "qwen36_decode_shape" not in source
