#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"


def _load_module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ck_run_v8_tests", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ck_run_v8 = _load_module()


def test_scratch_defaults_to_persistent_cache(
    tmp_path: Path, monkeypatch
) -> None:
    cache_home = tmp_path / "cache home"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.delenv("CK_V8_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", "/tmp/cached-before-test")

    scratch = ck_run_v8._configure_scratch_environment()

    assert scratch == (cache_home / "ck-engine-v8" / "tmp").resolve()
    assert os.environ["TMPDIR"] == str(scratch)
    assert tempfile.tempdir is None
    assert scratch.is_dir()


def test_scratch_honors_explicit_v8_override(
    tmp_path: Path, monkeypatch
) -> None:
    explicit = tmp_path / "compiler scratch"
    monkeypatch.setenv("CK_V8_TMPDIR", str(explicit))
    monkeypatch.setenv("TMPDIR", "/tmp/should-not-win")

    scratch = ck_run_v8._configure_scratch_environment()

    assert scratch == explicit.resolve()
    assert os.environ["TMPDIR"] == str(explicit.resolve())


def test_make_compiler_probe_uses_configured_scratch() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "CK_COMPILER_PROBE_DIR ?=" in makefile
    assert 'probe="$(CK_COMPILER_PROBE_DIR)/ck_cc_flag_test.$$$$.o"' in makefile
    assert "-o /tmp/ck_cc_flag_test.o" not in makefile


def test_refresh_manifest_circuit_snapshot_replaces_stale_graph_policy(
    tmp_path: Path, monkeypatch
) -> None:
    v8_root = tmp_path / "v8"
    circuits = v8_root / "circuits"
    circuits.mkdir(parents=True)
    current = {
        "name": "fixture",
        "version": 2,
        "kernels": {"attn_decode": "cache_aware_decode"},
    }
    (circuits / "fixture.json").write_text(json.dumps(current), encoding="utf-8")
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"model": "fixture"},
                "template": {
                    "name": "fixture",
                    "version": 1,
                    "kernels": {"attn": "stale_provider"},
                },
                "entries": [{"name": "weight", "offset": 0}],
            }
        ),
        encoding="utf-8",
    )
    original_entries = manifest_path_data(manifest_path)["entries"]
    monkeypatch.setattr(ck_run_v8, "V8_ROOT", v8_root)

    assert ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)
    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["template"] == current
    assert refreshed["entries"] == original_entries
    assert not ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)


def manifest_path_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bundle_stamp_rejects_changed_inputs_and_outputs(tmp_path: Path) -> None:
    source = tmp_path / "model_v8.c"
    output = tmp_path / "libmodel.so"
    stamp = tmp_path / ".ck_runtime_bundle.json"
    source.write_text("generated-v1", encoding="utf-8")
    output.write_bytes(b"runtime-v1")
    inputs = {"model_source": ck_run_v8._file_identity(source)}
    ck_run_v8._write_bundle_stamp(
        stamp,
        {
            "inputs": inputs,
            "outputs": {"libmodel.so": ck_run_v8._file_identity(output)},
        },
    )

    assert ck_run_v8._bundle_is_current(
        stamp, inputs=inputs, outputs={"libmodel.so": output}
    )

    source.write_text("generated-v2", encoding="utf-8")
    changed_inputs = {"model_source": ck_run_v8._file_identity(source)}
    assert not ck_run_v8._bundle_is_current(
        stamp, inputs=changed_inputs, outputs={"libmodel.so": output}
    )

    source.write_text("generated-v1", encoding="utf-8")
    output.write_bytes(b"runtime-corrupt")
    assert not ck_run_v8._bundle_is_current(
        stamp, inputs=inputs, outputs={"libmodel.so": output}
    )


def test_bundle_stamp_rejects_missing_or_malformed_stamp(tmp_path: Path) -> None:
    output = tmp_path / "libmodel.so"
    output.write_bytes(b"runtime")
    stamp = tmp_path / ".ck_runtime_bundle.json"
    inputs = {"schema": "fixture"}

    assert not ck_run_v8._bundle_is_current(
        stamp, inputs=inputs, outputs={"libmodel.so": output}
    )
    stamp.write_text("{broken", encoding="utf-8")
    assert not ck_run_v8._bundle_is_current(
        stamp, inputs=inputs, outputs={"libmodel.so": output}
    )


def test_sync_runtime_lib_replaces_same_size_stale_binary_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "runtime" / "library.so"
    source.write_bytes(b"new-runtime")
    destination.parent.mkdir()
    destination.write_bytes(b"old-runtime")

    ck_run_v8._sync_runtime_lib(source, destination, "fixture")

    assert destination.read_bytes() == b"new-runtime"
    assert not list(destination.parent.glob(".library.so.*.tmp"))


def test_sync_runtime_lib_refreshes_revalidated_identical_binary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "runtime" / "library.so"
    source.write_bytes(b"identical-runtime")
    destination.parent.mkdir()
    destination.write_bytes(source.read_bytes())
    os.utime(destination, ns=(1_000_000_000, 1_000_000_000))
    previous_mtime_ns = destination.stat().st_mtime_ns

    ck_run_v8._sync_runtime_lib(source, destination, "fixture")

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mtime_ns > previous_mtime_ns
    assert not list(destination.parent.glob(".library.so.*.tmp"))


def test_validate_runtime_bundle_reports_dynamic_loader_failure(
    tmp_path: Path, monkeypatch
) -> None:
    for name in (
        "libmodel.so",
        "libckernel_engine.so",
        "libckernel_tokenizer.so",
    ):
        (tmp_path / name).write_bytes(b"fixture")

    failure = subprocess.CompletedProcess(
        args=["python"],
        returncode=1,
        stdout="",
        stderr="OSError: undefined symbol: required_provider",
    )
    monkeypatch.setattr(ck_run_v8.subprocess, "run", lambda *args, **kwargs: failure)

    try:
        ck_run_v8._validate_runtime_bundle(tmp_path)
    except RuntimeError as exc:
        assert "undefined symbol: required_provider" in str(exc)
    else:
        raise AssertionError("invalid runtime bundle was accepted")


def test_validate_runtime_bundle_requires_all_three_libraries(
    tmp_path: Path,
) -> None:
    (tmp_path / "libmodel.so").write_bytes(b"fixture")

    try:
        ck_run_v8._validate_runtime_bundle(tmp_path)
    except RuntimeError as exc:
        assert "libckernel_engine.so" in str(exc)
        assert "libckernel_tokenizer.so" in str(exc)
    else:
        raise AssertionError("incomplete runtime bundle was accepted")
