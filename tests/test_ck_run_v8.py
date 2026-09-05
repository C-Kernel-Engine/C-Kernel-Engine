#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


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
    import build_ir_v8
    monkeypatch.setattr(build_ir_v8, "V8_ROOT", v8_root)

    assert ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)
    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["template"] == current
    assert refreshed["entries"] == original_entries
    assert not ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)


def manifest_path_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ir_fingerprint_includes_terminal_row_planner(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "weights_manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(ck_run_v8, "step_regenerate_kernel_registry", lambda **kwargs: None)
    monkeypatch.setattr(ck_run_v8, "_refresh_manifest_circuit_snapshot", lambda path: False)
    captured = []

    class FingerprintCaptured(Exception):
        pass

    def capture(paths):
        captured.extend(paths)
        raise FingerprintCaptured()

    monkeypatch.setattr(ck_run_v8, "_tree_identity", capture)
    with pytest.raises(FingerprintCaptured):
        ck_run_v8.step_build_ir(manifest, tmp_path)
    assert ck_run_v8.SCRIPTS_DIR / "plan_terminal_rows_v8.py" in captured


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


def test_sync_runtime_lib_refreshes_identical_rebuild_timestamp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "runtime" / "library.so"
    source.write_bytes(b"same-runtime")
    destination.parent.mkdir()
    destination.write_bytes(source.read_bytes())
    old_time_ns = 1_700_000_000_000_000_000
    new_time_ns = old_time_ns + 5_000_000_000
    os.utime(destination, ns=(old_time_ns, old_time_ns))
    os.utime(source, ns=(new_time_ns, new_time_ns))

    ck_run_v8._sync_runtime_lib(source, destination, "fixture")

    assert destination.read_bytes() == b"same-runtime"
    assert destination.stat().st_mtime_ns == new_time_ns
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


# --- Download integrity (nightly HF rate-limit resilience) -------------------

HTML_ERROR_PAGE = b"<!DOCTYPE html>\n<html><body>429 Too Many Requests</body></html>"


def _fake_hf_module(**overrides):
    import types

    def _unavailable(**kwargs):
        raise RuntimeError("huggingface_hub stubbed as unavailable")

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = overrides.get("hf_hub_download", _unavailable)
    module.snapshot_download = overrides.get("snapshot_download", _unavailable)
    return module


def _write_output_arg(cmd, payload: bytes):
    out_path = Path(cmd[cmd.index("-O") + 1] if "-O" in cmd else cmd[cmd.index("-o") + 1])
    out_path.write_bytes(payload)
    return subprocess.CompletedProcess(args=cmd, returncode=0)


def test_direct_hf_download_rejects_html_error_page(tmp_path: Path, monkeypatch) -> None:
    dst = tmp_path / "model.gguf"
    monkeypatch.setattr(
        ck_run_v8.shutil,
        "which",
        lambda name: None if name == "wget" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        ck_run_v8.subprocess,
        "run",
        lambda cmd, **kwargs: _write_output_arg(cmd, HTML_ERROR_PAGE),
    )

    assert not ck_run_v8._direct_hf_download_gguf("test/fake-repo-GGUF", "model.gguf", dst)
    assert not dst.exists()
    assert not dst.with_suffix(".gguf.part").exists()


def test_direct_hf_download_promotes_valid_gguf(tmp_path: Path, monkeypatch) -> None:
    dst = tmp_path / "model.gguf"
    payload = b"GGUF" + b"\x00" * 32
    monkeypatch.setattr(
        ck_run_v8.shutil,
        "which",
        lambda name: None if name == "wget" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        ck_run_v8.subprocess,
        "run",
        lambda cmd, **kwargs: _write_output_arg(cmd, payload),
    )

    assert ck_run_v8._direct_hf_download_gguf("test/fake-repo-GGUF", "model.gguf", dst)
    assert dst.read_bytes() == payload


def test_step_download_gguf_raises_distinct_error_after_429_retries(
    tmp_path: Path, monkeypatch
) -> None:
    def _rate_limited(**kwargs):
        raise RuntimeError("HTTP Error 429 thrown while requesting HEAD https://huggingface.co/x")

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_rate_limited))
    monkeypatch.setattr(ck_run_v8, "_direct_hf_download_gguf", lambda *args: False)
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [tmp_path / "cache"])

    try:
        ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", tmp_path / "cache")
    except ck_run_v8.V8DownloadError as exc:
        message = str(exc)
        assert ck_run_v8.DOWNLOAD_ERROR_MARKER in message
        assert "429" in message
        assert "rate limit" in message
    else:
        raise AssertionError("429-exhausted download did not raise V8DownloadError")
    assert not (tmp_path / "cache" / "test--fake-repo-GGUF" / "fake-model.gguf").exists()


def test_step_download_gguf_rejects_non_gguf_payload(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"

    def _error_page(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(HTML_ERROR_PAGE)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_error_page))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])
    # The error-page case is transient and retried; disable outer retries so
    # this test stays fast (retry behavior is covered separately below).
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "0")

    try:
        ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)
    except ck_run_v8.V8DownloadError as exc:
        assert ck_run_v8.DOWNLOAD_ERROR_MARKER in str(exc)
    else:
        raise AssertionError("HTML error page payload was accepted as a GGUF")
    assert not (cache_dir / "test--fake-repo-GGUF" / "fake-model.gguf").exists()


def test_step_download_gguf_moves_valid_payload_into_place(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    payload = b"GGUF" + b"\x01" * 64

    def _valid_download(**kwargs):
        staging = Path(kwargs["local_dir"]) / ".staging" / kwargs["filename"]
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_bytes(payload)
        return str(staging)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_valid_download))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])

    result = ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)

    assert result == cache_dir / "test--fake-repo-GGUF" / "fake-model.gguf"
    assert result.read_bytes() == payload


def test_step_download_gguf_discards_invalid_cached_payload(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "test--fake-repo-GGUF" / "fake-model.gguf"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(HTML_ERROR_PAGE)
    payload = b"GGUF" + b"\x02" * 64

    def _valid_download(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_valid_download))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])

    result = ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)

    assert result.read_bytes() == payload


def test_ensure_tokenizer_files_drops_invalid_json_payload(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"

    def _error_page(**kwargs):
        target = Path(kwargs["local_dir"]) / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(HTML_ERROR_PAGE)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_error_page))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [tmp_path / "empty-cache"])
    # The error-page case is transient and retried; disable outer retries so
    # this test stays fast (retry behavior is covered separately below).
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "0")

    with pytest.raises(ck_run_v8.V8DownloadError, match=ck_run_v8.DOWNLOAD_ERROR_MARKER):
        ck_run_v8.ensure_tokenizer_files("test/fake-repo-GGUF", work_dir)

    assert not (work_dir / "tokenizer.json").exists()


def test_ensure_tokenizer_files_discards_invalid_cached_json(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "test--fake-repo-GGUF" / "tokenizer.json"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(HTML_ERROR_PAGE)

    def _valid_json(**kwargs):
        target = Path(kwargs["local_dir"]) / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"model": {"type": "BPE"}}', encoding="utf-8")
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_valid_json))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])

    ck_run_v8.ensure_tokenizer_files("test/fake-repo-GGUF", work_dir)

    assert not cached.exists()
    assert ck_run_v8._validate_json_file(work_dir / "tokenizer.json")


def test_ensure_tokenizer_files_keeps_valid_json_payload(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"

    def _valid_json(**kwargs):
        target = Path(kwargs["local_dir"]) / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"model": {"type": "BPE"}}', encoding="utf-8")
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_valid_json))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [tmp_path / "empty-cache"])

    ck_run_v8.ensure_tokenizer_files("test/fake-repo-GGUF", work_dir)

    assert json.loads((work_dir / "tokenizer.json").read_text(encoding="utf-8")) == {
        "model": {"type": "BPE"}
    }


# --- Rate-limit-aware outer retries -------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _FakeHttpError(RuntimeError):
    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        super().__init__(
            f"HTTP Error {status_code} thrown while requesting HEAD https://huggingface.co/x"
        )
        self.response = _FakeResponse(status_code, headers)


def _record_sleeps(monkeypatch) -> list:
    sleeps: list[float] = []
    monkeypatch.setattr(ck_run_v8.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_step_download_gguf_retries_429_honoring_retry_after(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    payload = b"GGUF" + b"\x02" * 64
    calls = {"count": 0}

    def _flaky_download(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise _FakeHttpError(429, headers={"Retry-After": "120"})
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_flaky_download))
    monkeypatch.setattr(ck_run_v8, "_direct_hf_download_gguf", lambda *args: False)
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "3")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "15")
    sleeps = _record_sleeps(monkeypatch)

    result = ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)

    assert calls["count"] == 3
    assert result.read_bytes() == payload
    # Retry-After (120s) exceeds the computed backoff (15/30 + jitter <= 35s),
    # so the server hint wins on both waits.
    assert sleeps == [120.0, 120.0]


def test_step_download_gguf_404_fails_immediately_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    calls = {"count": 0}

    def _missing(**kwargs):
        calls["count"] += 1
        raise _FakeHttpError(404)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_missing))
    monkeypatch.setattr(ck_run_v8, "_direct_hf_download_gguf", lambda *args: False)
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "3")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "0")
    sleeps = _record_sleeps(monkeypatch)

    try:
        ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "absent.gguf", cache_dir)
    except ck_run_v8.V8DownloadError as exc:
        assert ck_run_v8.DOWNLOAD_ERROR_MARKER in str(exc)
        assert "404" in str(exc)
    else:
        raise AssertionError("404 download did not fail")
    assert calls["count"] == 1
    assert sleeps == []


def test_step_download_gguf_exhausted_retries_raise_marked_error(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    calls = {"count": 0}

    def _always_limited(**kwargs):
        calls["count"] += 1
        raise _FakeHttpError(429)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_always_limited))
    monkeypatch.setattr(ck_run_v8, "_direct_hf_download_gguf", lambda *args: False)
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "2")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "0.01")
    sleeps = _record_sleeps(monkeypatch)

    try:
        ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)
    except ck_run_v8.V8DownloadError as exc:
        message = str(exc)
        assert ck_run_v8.DOWNLOAD_ERROR_MARKER in message
        assert "429" in message
        assert "3/3" in message
    else:
        raise AssertionError("exhausted 429 retries did not raise V8DownloadError")
    assert calls["count"] == 3
    assert len(sleeps) == 2


def test_download_retry_backoff_bounds_with_jitter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "3")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "15")
    sleeps = _record_sleeps(monkeypatch)

    def _always_limited():
        raise _FakeHttpError(503)

    try:
        ck_run_v8._run_with_download_retries("test/fake-repo-GGUF", "fake-model.gguf", _always_limited)
    except ck_run_v8.V8DownloadError:
        pass
    else:
        raise AssertionError("persistent 503 did not raise V8DownloadError")

    assert len(sleeps) == 3
    for index, sleep_s in enumerate(sleeps):
        base = 15.0 * (2 ** index)
        assert base <= sleep_s <= base + 5.0


def test_download_retry_after_overrides_computed_backoff(monkeypatch) -> None:
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "1")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "15")
    sleeps = _record_sleeps(monkeypatch)
    attempts = {"count": 0}

    def _limited_once():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _FakeHttpError(429, headers={"Retry-After": "90"})
        return "ok"

    assert ck_run_v8._run_with_download_retries("test/repo", "file.gguf", _limited_once) == "ok"
    assert sleeps == [90.0]


def test_retry_after_ignores_unparseable_values() -> None:
    error = _FakeHttpError(429, headers={"Retry-After": "soon"})
    assert ck_run_v8._retry_after_seconds(error) is None
    assert ck_run_v8._retry_after_seconds(_FakeHttpError(429, headers={"Retry-After": "45"})) == 45.0
    assert ck_run_v8._retry_after_seconds(_FakeHttpError(429)) is None


def test_step_download_gguf_retries_error_page_then_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    payload = b"GGUF" + b"\x03" * 64
    calls = {"count": 0}

    def _error_page_then_valid(**kwargs):
        calls["count"] += 1
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(HTML_ERROR_PAGE if calls["count"] == 1 else payload)
        return str(target)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_error_page_then_valid)
    )
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [cache_dir])
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "3")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "0.01")
    sleeps = _record_sleeps(monkeypatch)

    result = ck_run_v8.step_download_gguf("test/fake-repo-GGUF", "fake-model.gguf", cache_dir)

    assert calls["count"] == 2
    assert result.read_bytes() == payload
    assert len(sleeps) == 1


def test_ensure_tokenizer_files_retries_429_then_keeps_valid_json(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    calls = {"count": 0}

    def _flaky_tokenizer(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _FakeHttpError(429)
        target = Path(kwargs["local_dir"]) / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"model": {"type": "BPE"}}', encoding="utf-8")
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf_module(hf_hub_download=_flaky_tokenizer))
    monkeypatch.setattr(ck_run_v8, "_cache_roots", lambda: [tmp_path / "empty-cache"])
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRIES", "3")
    monkeypatch.setenv("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "0.01")
    sleeps = _record_sleeps(monkeypatch)

    ck_run_v8.ensure_tokenizer_files("test/fake-repo-GGUF", work_dir)

    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert (work_dir / "tokenizer.json").exists()
