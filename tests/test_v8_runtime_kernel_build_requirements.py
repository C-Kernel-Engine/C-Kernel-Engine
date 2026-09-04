import json
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))

import ck_run_v8  # noqa: E402


def _write_call(
    root: Path,
    kernel_id: str,
    owner: str,
    source_file: str,
    *,
    op: str | None = None,
    function: str | None = None,
    args: list[dict] | None = None,
) -> None:
    if args is None:
        args = [
            {"name": "k_dst", "source": "output:k_dst"},
            {"name": "k_src", "source": "activation:k_src"},
            {"name": "v_dst", "source": "output:v_dst"},
            {"name": "v_src", "source": "activation:v_src"},
            {"name": "size", "source": "dim:_kv_copy_bytes"},
        ]
    (root / "lowered_decode_call.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": op or kernel_id,
                        "function": function or kernel_id,
                        "args": args,
                        "errors": [],
                        "call_abi": {
                            "version": 1,
                            "kernel_id": kernel_id,
                            "owner": owner,
                            "source_file": source_file,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_registry(root: Path, entries: list[dict]) -> Path:
    path = root / "registry.json"
    path.write_text(json.dumps({"kernels": entries}), encoding="utf-8")
    return path


def test_selected_onednn_provider_enables_engine_feature(tmp_path: Path) -> None:
    _write_call(tmp_path, "exact_gemm", "kernel_map", "exact_gemm.json")
    registry = _write_registry(
        tmp_path,
        [
            {
                "id": "exact_gemm",
                "impl": {
                    "variants": [
                        {"compile_flags": ["-mavx512f", "-DUSE_ONEDNN"]}
                    ]
                },
            }
        ],
    )
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), mock.patch.object(
        ck_run_v8, "_onednn_make_args", return_value=["USE_ONEDNN=1"]
    ):
        assert ck_run_v8._selected_kernel_build_args(tmp_path) == ["USE_ONEDNN=1"]


def test_codegen_owned_kv_copy_cannot_bypass_the_registry(tmp_path: Path) -> None:
    _write_call(
        tmp_path,
        "kv_cache_batch_copy",
        "legacy_compatibility",
        "kernel_bindings*.json",
        op="kv_cache_batch_copy",
    )
    registry = _write_registry(tmp_path, [])
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), pytest.raises(
        RuntimeError, match="absent from the registry"
    ):
        ck_run_v8._selected_kernel_build_args(tmp_path)


def test_registered_copy_last_logits_is_not_a_helper_exception(tmp_path: Path) -> None:
    _write_call(
        tmp_path,
        "copy_last_logits",
        "kernel_map",
        "copy_last_logits.json",
        op="copy_last_logits",
    )
    registry = _write_registry(
        tmp_path,
        [{"id": "copy_last_logits", "impl": {"variants": []}}],
    )
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry):
        assert ck_run_v8._selected_kernel_build_args(tmp_path) == []


def test_unregistered_copy_last_logits_fails_closed(tmp_path: Path) -> None:
    _write_call(
        tmp_path,
        "copy_last_logits",
        "legacy_compatibility",
        "kernel_bindings*.json",
        op="copy_last_logits",
    )
    registry = _write_registry(tmp_path, [])
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), pytest.raises(
        RuntimeError, match="absent from the registry"
    ):
        ck_run_v8._selected_kernel_build_args(tmp_path)


@pytest.mark.parametrize(
    "kernel_id,owner,source_file",
    [
        ("unknown_provider", "kernel_map", "unknown.json"),
        ("kv_cache_batch_copy", "kernel_map", "wrong.json"),
    ],
)
def test_other_non_registry_selections_fail_closed(
    tmp_path: Path, kernel_id: str, owner: str, source_file: str
) -> None:
    _write_call(tmp_path, kernel_id, owner, source_file)
    registry = _write_registry(tmp_path, [])
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), pytest.raises(
        RuntimeError, match="absent from the registry"
    ):
        ck_run_v8._selected_kernel_build_args(tmp_path)


@pytest.mark.parametrize(
    "op,owner,source_file",
    [
        ("not_kv_cache_copy", "legacy_compatibility", "kernel_bindings*.json"),
        ("kv_cache_batch_copy", "kernel_map", "kernel_bindings*.json"),
        ("kv_cache_batch_copy", "legacy_compatibility", "not_bindings.json"),
    ],
)
def test_mismatched_identity_cannot_claim_generated_kv_helper_exemption(
    tmp_path: Path, op: str, owner: str, source_file: str
) -> None:
    _write_call(
        tmp_path,
        "kv_cache_batch_copy",
        owner,
        source_file,
        op=op,
    )
    registry = _write_registry(tmp_path, [])
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), pytest.raises(
        RuntimeError, match="absent from the registry"
    ):
        ck_run_v8._selected_kernel_build_args(tmp_path)


@pytest.mark.parametrize(
    "function,args,abi_version",
    [
        ("memcpy", None, 1),
        (
            "kv_cache_batch_copy",
            [
                {"name": "k_dst", "source": "output:k_dst"},
                {"name": "k_src", "source": "activation:not_k_src"},
                {"name": "v_dst", "source": "output:v_dst"},
                {"name": "v_src", "source": "activation:v_src"},
                {"name": "size", "source": "dim:_kv_copy_bytes"},
            ],
            1,
        ),
        (
            "kv_cache_batch_copy",
            [
                {"name": "k_dst", "source": "output:k_dst"},
                {"name": "k_src", "source": "activation:k_src"},
                {"name": "v_dst", "source": "output:v_dst"},
                {"name": "v_src", "source": "activation:v_src"},
                {"name": "size", "source": "dim:_kv_copy_bytes"},
                {"name": "size", "source": "dim:_kv_copy_bytes"},
            ],
            1,
        ),
        ("kv_cache_batch_copy", None, 2),
    ],
)
def test_non_kv_call_shape_cannot_claim_generated_kv_helper_exemption(
    tmp_path: Path,
    function: str,
    args: list[dict] | None,
    abi_version: int,
) -> None:
    _write_call(
        tmp_path,
        "kv_cache_batch_copy",
        "legacy_compatibility",
        "kernel_bindings*.json",
        function=function,
        args=args,
    )
    document = json.loads((tmp_path / "lowered_decode_call.json").read_text())
    document["operations"][0]["call_abi"]["version"] = abi_version
    (tmp_path / "lowered_decode_call.json").write_text(json.dumps(document))
    registry = _write_registry(tmp_path, [])
    with mock.patch.object(ck_run_v8, "KERNEL_REGISTRY_PATH", registry), pytest.raises(
        RuntimeError, match="absent from the registry"
    ):
        ck_run_v8._selected_kernel_build_args(tmp_path)
