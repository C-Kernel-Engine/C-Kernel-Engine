from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_whisper_v8.py"
UNIFIED_SCRIPT = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"
PYTORCH_PARITY_SCRIPT = (
    ROOT / "version" / "v8" / "scripts" / "compare_whisper_e2e_pytorch_v8.py"
)
FRONTEND_XRAY_SCRIPT = (
    ROOT
    / "version"
    / "v8"
    / "scripts"
    / "compare_whisper_frontend_pytorch_v8.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("run_whisper_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unified_module():
    spec = importlib.util.spec_from_file_location("ck_run_v8", UNIFIED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pytorch_parity_module():
    spec = importlib.util.spec_from_file_location(
        "compare_whisper_e2e_pytorch_v8", PYTORCH_PARITY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frontend_xray_module():
    spec = importlib.util.spec_from_file_location(
        "compare_whisper_frontend_pytorch_v8", FRONTEND_XRAY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whisper_runner_uses_generated_frontend_and_forced_prefix_is_stable() -> None:
    runner = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ck_model_prepare_audio_wav_window" in source
    assert "ck_model_run_encoder" in source
    assert "audio_resample_windowed_sinc_f32" not in source
    assert "audio_stft_power_fft400_f32" not in source
    assert "audio_whisper_log_mel_from_power_reference_f32" not in source
    generation = {
        "decoder_start_token_id": 50258,
        "lang_to_id": {"<|en|>": 50259},
        "task_to_id": {"transcribe": 50359},
        "no_timestamps_token_id": 50363,
    }
    assert runner.forced_decoder_prefix(generation, "en", "transcribe") == [
        50258,
        50259,
        50359,
        50363,
    ]
    assert runner.forced_decoder_prefix(
        generation, "en", "transcribe", timestamps=True
    ) == [50258, 50259, 50359]


def test_whisper_persistent_workers_are_default_with_control_fallback() -> None:
    runner = _module()
    required = [
        "run",
        "--encoder-run-dir",
        "/encoder",
        "--decoder-run-dir",
        "/decoder",
        "--wav",
        "/audio.wav",
    ]
    defaults = runner._parser().parse_args(required)
    assert defaults.worker_lifecycle == "persistent"
    assert defaults.worker_timeout_seconds == 600.0
    assert (
        runner._parser().parse_args(
            [*required, "--worker-lifecycle", "per-window"]
        ).worker_lifecycle
        == "per-window"
    )


def test_whisper_runner_measures_frontend_and_encoder_separately() -> None:
    runner = _module()
    calls: list[str] = []

    def prepare(*_args) -> int:
        calls.append("frontend")
        return 0

    def encode() -> int:
        calls.append("encoder")
        return 0

    model = SimpleNamespace(
        ck_model_prepare_audio_wav_window=prepare,
        ck_model_run_encoder=encode,
    )
    wav = np.zeros(32, dtype=np.uint8)
    with mock.patch.object(
        runner.time,
        "perf_counter",
        side_effect=[1.0, 1.25, 2.0, 2.75],
    ):
        frontend_seconds, encoder_seconds = runner._run_audio_encoder_window(
            model, wav, 0, runner.CKAudioWavInfo()
        )

    assert calls == ["frontend", "encoder"]
    assert frontend_seconds == 0.25
    assert encoder_seconds == 0.75
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"frontend_seconds": sum(' in source
    assert '"encoder_engine_sha256": _sha256(' in source
    assert '"decoder_engine_sha256": _sha256(' in source
    assert '"max_tokens_per_window": int(args.max_tokens)' in source


def test_whisper_cached_frontend_slices_and_zero_pads_exactly() -> None:
    runner = _module()
    full = np.arange(16, dtype=np.float32).reshape(2, 8)
    output = np.full((2, 5), -1.0, dtype=np.float32)

    valid = runner._copy_cached_feature_window(
        full,
        output,
        window_start_frame=4,
        hop_length=2,
    )
    assert valid == 5
    np.testing.assert_array_equal(output, full[:, 2:7])

    valid = runner._copy_cached_feature_window(
        full,
        output,
        window_start_frame=12,
        hop_length=2,
    )
    assert valid == 2
    np.testing.assert_array_equal(output[:, :2], full[:, 6:8])
    np.testing.assert_array_equal(output[:, 2:], np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="hop-aligned"):
        runner._copy_cached_feature_window(
            full,
            output,
            window_start_frame=3,
            hop_length=2,
        )


def test_whisper_frontend_cache_capacity_accounts_for_outputs() -> None:
    runner = _module()
    config = {
        "audio_feature_channels": 80,
        "context_length": 1500,
        "embed_dim": 512,
    }
    assert runner._frontend_cache_required_bytes(config, 3000) == (
        80 * 3000 * 4 + 1500 * 512 * 4 + 8 * 1024 * 1024
    )


def test_whisper_frontend_cache_quota_failure_requests_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module()

    def fail_allocation(*_args) -> None:
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(runner.os, "posix_fallocate", fail_allocation)
    reason = runner._probe_cache_capacity(tmp_path, 128 * 1024 * 1024)
    assert reason is not None
    assert "cannot reserve" in reason
    assert not (tmp_path / ".cke-whisper-capacity-probe").exists()


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (78, "rebuild to enable it"),
        (79, "could not be stored"),
    ],
)
def test_whisper_frontend_worker_expected_failures_request_fallback(
    returncode: int, expected: str
) -> None:
    runner = _module()
    reason = runner._frontend_reuse_worker_failure(returncode)
    assert reason is not None
    assert expected in reason
    with pytest.raises(ValueError, match="unexpected frontend worker"):
        runner._frontend_reuse_worker_failure(1)


def test_whisper_workers_do_not_compete_with_numpy_blas_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module()
    monkeypatch.setenv("CK_NUM_THREADS", "20")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "28")
    monkeypatch.setenv("MKL_NUM_THREADS", "28")

    worker_env = runner._worker_environment()

    assert worker_env["CK_NUM_THREADS"] == "20"
    assert worker_env["OPENBLAS_NUM_THREADS"] == "1"
    assert worker_env["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "28"
    assert os.environ["MKL_NUM_THREADS"] == "28"


def test_whisper_persistent_worker_initializes_and_frees_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module()
    requests = iter(
        [
            {"command": "run", "args": {"value": 1}},
            {"command": "run", "args": {"value": 2}},
            {"command": "close"},
        ]
    )

    class Connection:
        def __init__(self) -> None:
            self.responses: list[dict[str, object]] = []

        def recv(self) -> dict[str, object]:
            return next(requests)

        def send(self, response: dict[str, object]) -> None:
            self.responses.append(response)

        def close(self) -> None:
            pass

    class Model:
        def __init__(self) -> None:
            self.free_count = 0

        def ck_model_free(self) -> None:
            self.free_count += 1

    connection = Connection()
    model = Model()
    observed: list[int] = []
    monkeypatch.setattr(runner, "_apply_worker_affinity", lambda: {"cpus": [0]})
    monkeypatch.setattr(
        runner, "_open_encoder_model", lambda _run_dir: (model, 0.125)
    )
    monkeypatch.setattr(
        runner,
        "_encoder_worker",
        lambda args, **_kwargs: observed.append(args.value),
    )

    runner._persistent_worker_main(connection, "encoder", str(tmp_path), {})

    assert observed == [1, 2]
    assert model.free_count == 1
    assert connection.responses == [
        {
            "status": "ready",
            "role": "encoder",
            "model_init_seconds": 0.125,
            "execution_topology": {"cpus": [0]},
        },
        {"status": "ok", "role": "encoder"},
        {"status": "ok", "role": "encoder"},
        {"status": "closed", "role": "encoder"},
    ]


def test_whisper_decoder_resets_before_rebinding_encoder_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module()
    run_dir = tmp_path / "decoder"
    run_dir.mkdir()
    (run_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "decoder_start_token_id": 1,
                "lang_to_id": {"<|en|>": 2},
                "task_to_id": {"transcribe": 3},
                "no_timestamps_token_id": 4,
                "eos_token_id": 5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    encoder_output = tmp_path / "encoder.npy"
    np.save(encoder_output, np.zeros((2, 3), dtype=np.float32))
    report = tmp_path / "decoder.json"
    calls: list[str] = []

    class Model:
        def ck_model_kv_cache_reset(self) -> None:
            calls.append("reset")

        def ck_model_set_encoder_memory(self, *_args) -> int:
            calls.append("bind")
            return 0

        def ck_model_embed_tokens(self, *_args) -> int:
            calls.append("prefill")
            return 0

        def ck_model_get_vocab_size(self) -> int:
            return 6

        def ck_model_get_logits(self):
            return (runner.ctypes.c_float * 6)(0, 0, 0, 0, 0, 1)

    class Tokenizer:
        @staticmethod
        def from_file(_path: str):
            return Tokenizer()

        def decode(self, *_args, **_kwargs) -> str:
            return ""

    monkeypatch.setattr(runner, "_require_artifact", lambda _path: None)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    args = SimpleNamespace(
        decoder_run_dir=run_dir,
        encoder_output=encoder_output,
        language="en",
        task="transcribe",
        max_tokens=0,
        timestamps=False,
        worker_report=report,
    )

    runner._decoder_worker(args, model=Model(), execution_topology={})

    assert calls[:3] == ["reset", "bind", "prefill"]


def test_whisper_hybrid_topology_selects_smt_capable_cores(
    tmp_path: Path,
) -> None:
    runner = _module()
    siblings = {
        0: "0-1\n",
        1: "0-1\n",
        2: "2-3\n",
        3: "2-3\n",
        4: "4\n",
        5: "5\n",
    }
    for cpu, value in siblings.items():
        path = tmp_path / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(value, encoding="ascii")
    assert runner._hybrid_performance_cpus(
        set(siblings), sysfs_root=tmp_path
    ) == [0, 1, 2, 3]


def test_whisper_uniform_topology_does_not_restrict_affinity(
    tmp_path: Path,
) -> None:
    runner = _module()
    for cpu, value in {0: "0-1\n", 1: "0-1\n", 2: "2-3\n", 3: "2-3\n"}.items():
        path = tmp_path / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(value, encoding="ascii")
    assert runner._hybrid_performance_cpus(
        {0, 1, 2, 3}, sysfs_root=tmp_path
    ) is None


def test_whisper_explicit_thread_count_remains_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module()
    monkeypatch.setenv("CK_NUM_THREADS", "7")
    monkeypatch.setattr(
        runner, "_hybrid_performance_cpus", lambda allowed: [0, 1, 2, 3]
    )
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4})
    env = runner._worker_environment(
        {"audio_runtime_topology_policy": "performance_core_smt_on_hybrid"}
    )
    assert env["CK_AUDIO_WORKER_CPUS"] == "0,1,2,3"
    assert env["CK_NUM_THREADS"] == "7"


def test_whisper_frontend_xray_metrics_are_json_serializable() -> None:
    xray = _frontend_xray_module()
    reference = np.zeros((2, 3), dtype=np.float32)
    actual = reference.copy()
    actual[1, 2] = np.float32(0.25)
    metrics = xray._metrics(reference, actual)
    assert metrics["worst_coordinate"] == [1, 2]
    assert metrics["max_abs"] == 0.25
    json.dumps(metrics)


def test_whisper_frontend_xray_resolves_scalar_call_ir_identity() -> None:
    xray = _frontend_xray_module()
    execution = xray._resolved_execution(
        {
            "function": "audio_feature_window",
            "resolved_contract": {
                "kernel_id": "audio.feature.window",
                "function": "audio_feature_window",
                "resolved_contract_id": "audio.feature.window.fp32",
            },
        }
    )
    assert execution == {
        "kernel_id": "audio.feature.window",
        "function": "audio_feature_window",
        "resolved_contract_id": "audio.feature.window.fp32",
    }
    assert all(
        value is None or isinstance(value, str)
        for value in execution.values()
    )


def test_whisper_timestamp_contract_enforces_initial_pair_and_order() -> None:
    runner = _module()
    generation = {
        "no_timestamps_token_id": 10,
        "eos_token_id": 9,
        "max_initial_timestamp_index": 2,
    }
    logits = np.arange(16, dtype=np.float32)

    initial = runner.apply_timestamp_logits_contract(
        logits, [], generation
    )
    assert np.all(np.isneginf(initial[:11]))
    assert np.all(np.isfinite(initial[11:14]))
    assert np.all(np.isneginf(initial[14:]))

    after_open = runner.apply_timestamp_logits_contract(
        logits, [12], generation
    )
    assert np.all(np.isfinite(after_open[:10]))
    assert np.isneginf(after_open[10])
    assert np.all(np.isneginf(after_open[11:]))

    after_text = runner.apply_timestamp_logits_contract(
        logits, [12, 4], generation
    )
    assert np.all(np.isneginf(after_text[11:13]))
    assert np.all(np.isfinite(after_text[13:]))

    after_close = runner.apply_timestamp_logits_contract(
        logits, [12, 4, 13], generation
    )
    assert np.all(np.isneginf(after_close[10:13]))
    assert np.all(np.isfinite(after_close[13:]))


def test_whisper_timestamp_contract_uses_aggregate_probability() -> None:
    runner = _module()
    generation = {
        "no_timestamps_token_id": 4,
        "eos_token_id": 3,
        "max_initial_timestamp_index": None,
    }
    logits = np.asarray(
        [2.0, 0.0, 0.0, 0.0, -5.0, 1.5, 1.5, 1.5]
    )
    result = runner.apply_timestamp_logits_contract(
        logits, [5, 1], generation
    )
    assert np.all(np.isneginf(result[:5]))
    assert np.isfinite(result[6])


def test_whisper_long_audio_window_plan_is_complete_and_non_overlapping() -> None:
    runner = _module()
    assert runner.plan_audio_windows(480000, 16000, 16000, 480000) == [
        (0, 480000)
    ]
    assert runner.plan_audio_windows(960001, 16000, 16000, 480000) == [
        (0, 480000),
        (480000, 960000),
        (960000, 960001),
    ]
    with pytest.raises(ValueError, match="globally phased resampling"):
        runner.plan_audio_windows(1_440_001, 48000, 16000, 480000)


def test_whisper_segment_timestamps_are_offset_to_the_full_audio() -> None:
    runner = _module()
    generation = {"no_timestamps_token_id": 50363}
    assert runner.global_timestamp_events(
        [50364, 400, 50414], generation, 30.0
    ) == [
        {
            "token_id": 50364,
            "local_seconds": 0.0,
            "global_seconds": 30.0,
        },
        {
            "token_id": 50414,
            "local_seconds": 1.0,
            "global_seconds": 31.0,
        },
    ]


def test_whisper_timestamp_seek_matches_reference_segment_rules() -> None:
    runner = _module()
    generation = {"no_timestamps_token_id": 50363}
    assert runner.timestamp_seek_consumed_frames(
        [50364, 400, 50914, 50914, 500, 51464, 51464],
        generation,
        16000,
        480000,
    ) == 352000
    assert runner.timestamp_seek_consumed_frames(
        [50364, 400, 50744, 50744, 500, 50894],
        generation,
        16000,
        176000,
    ) == 176000
    assert runner.timestamp_seek_consumed_frames(
        [50364, 400, 50894],
        generation,
        16000,
        176000,
    ) == 176000


def test_whisper_timestamp_window_consumes_only_timestamp_sized_tail() -> None:
    runner = _module()
    source_frames = 4_800_000
    assert runner.consume_timestamp_sized_tail(
        source_frames - 640, source_frames, 16000
    ) == source_frames
    assert runner.consume_timestamp_sized_tail(
        source_frames - 1920, source_frames, 16000
    ) == source_frames - 1920


def test_whisper_timestamp_contract_matches_transformers_masks() -> None:
    torch = pytest.importorskip("torch")
    generation_module = pytest.importorskip(
        "transformers.generation.logits_process"
    )
    runner = _module()
    generation = {
        "no_timestamps_token_id": 10,
        "eos_token_id": 9,
        "max_initial_timestamp_index": 2,
    }
    config = SimpleNamespace(
        **generation,
        bos_token_id=0,
        _detect_timestamp_from_logprob=True,
    )
    processor = generation_module.WhisperTimeStampLogitsProcessor(
        config, begin_index=3
    )
    prefix = [1, 2, 3]
    logits = np.asarray(
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, -5.0, 1.5, 1.5, 1.5, 1.0, 0.5],
        dtype=np.float32,
    )
    for generated in ([], [12], [12, 4], [12, 4, 13]):
        actual = runner.apply_timestamp_logits_contract(
            logits, generated, generation
        )
        expected = processor(
            torch.tensor([prefix + generated]),
            torch.from_numpy(logits.copy()).reshape(1, -1),
        )[0].numpy()
        np.testing.assert_array_equal(actual, expected)


def test_unified_v8_cli_owns_the_public_audio_command() -> None:
    source = UNIFIED_SCRIPT.read_text(encoding="utf-8")
    assert 'subparsers.add_parser(\n        "audio"' in source
    assert "run_audio_pipeline(args)" in source
    completed = subprocess.run(
        [sys.executable, str(UNIFIED_SCRIPT), "audio", "--help"],
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--encoder-run-dir" in completed.stdout
    assert "--decoder-run-dir" in completed.stdout
    assert "--wav" in completed.stdout
    assert "model" in completed.stdout
    assert "--force-convert" in completed.stdout
    assert "--timestamps" in completed.stdout


def test_whisper_pytorch_target_exposes_timestamp_gate() -> None:
    source = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = source.split("test-whisper-pytorch-e2e-auto:", 1)[1]
    target = target.split("\nnightly-parity:", 1)[0]
    assert "CK_WHISPER_TIMESTAMPS" in target
    assert 'timestamp_arg="--timestamps"' in target


def test_unified_audio_checkpoint_builds_distinct_generic_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unified = _unified_module()
    checkpoint = tmp_path / "whisper"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    run_dir = tmp_path / "runtime"
    calls: list[tuple[Path, Path, str, bool, bool, str]] = []

    def fake_build_role(
        checkpoint_dir: Path,
        role_dir: Path,
        role: str,
        *,
        force_convert: bool,
        force_compile: bool,
        linear_weight_dtype: str = "preserve",
    ) -> None:
        calls.append(
            (
                checkpoint_dir,
                role_dir,
                role,
                force_convert,
                force_compile,
                linear_weight_dtype,
            )
        )

    monkeypatch.setattr(unified, "_build_whisper_role", fake_build_role)
    monkeypatch.setattr(
        unified, "_is_safetensors_checkpoint_dir", lambda path: True
    )
    encoder, decoder = unified.step_build_whisper_runtimes(
        str(checkpoint),
        run_dir=run_dir,
        force_download=False,
        force_convert=True,
        force_compile=False,
        encoder_linear_weight_dtype="fp16",
    )
    assert encoder == run_dir / "encoder"
    assert decoder == run_dir / "decoder"
    assert [row[2] for row in calls] == ["encoder", "decoder"]
    assert all(row[0] == checkpoint for row in calls)
    assert all(row[3] is True and row[4] is False for row in calls)
    assert [row[5] for row in calls] == ["fp16", "preserve"]


def test_whisper_build_identity_includes_linear_weight_policy(tmp_path: Path) -> None:
    unified = _unified_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    fp32 = unified._whisper_build_inputs(
        checkpoint, "encoder", linear_weight_dtype="preserve"
    )
    fp16 = unified._whisper_build_inputs(
        checkpoint, "encoder", linear_weight_dtype="fp16"
    )
    assert fp32["linear_weight_dtype"] == "preserve"
    assert fp16["linear_weight_dtype"] == "fp16"
    assert fp32 != fp16


def test_whisper_checkpoint_identity_changes_with_source(
    tmp_path: Path,
) -> None:
    unified = _unified_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    weights = checkpoint / "model.safetensors"
    config.write_text('{"d_model": 384}\n', encoding="utf-8")
    weights.write_bytes(b"weights")
    first = unified._whisper_checkpoint_identity(checkpoint)
    config.write_text('{"d_model": 512}\n', encoding="utf-8")
    second = unified._whisper_checkpoint_identity(checkpoint)
    assert first["model.safetensors"] == second["model.safetensors"]
    assert first["config.json"] != second["config.json"]


def test_whisper_pytorch_parity_reports_first_token_or_length_difference() -> None:
    parity = _pytorch_parity_module()
    source = PYTORCH_PARITY_SCRIPT.read_text(encoding="utf-8")
    assert '"truncation": False' in source
    assert '"return_attention_mask": True' in source
    assert "inputs.attention_mask" in source
    assert parity.first_token_difference([1, 2, 3], [1, 2, 3]) is None
    assert parity.first_token_difference([1, 9, 3], [1, 2, 3]) == {
        "index": 1,
        "subject": 9,
        "oracle": 2,
    }
    assert parity.first_token_difference([1, 2], [1, 2, 3]) == {
        "index": 2,
        "subject": -1,
        "oracle": 3,
    }


def test_whisper_pytorch_parity_fails_closed_on_stale_runtime(
    tmp_path: Path,
) -> None:
    parity = _pytorch_parity_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"whisper"}\n', encoding="utf-8"
    )
    identity = parity.checkpoint_identity(checkpoint)
    runtime = tmp_path / "encoder"
    runtime.mkdir()
    stamp = runtime / ".ck-whisper-runtime.json"
    stamp.write_text(
        json.dumps(
            {
                "inputs": {
                    "role": "encoder",
                    "checkpoint": identity,
                }
            }
        ),
        encoding="utf-8",
    )
    parity.validate_runtime_checkpoint(runtime, identity, "encoder")
    changed = dict(identity)
    changed["config.json"] = dict(changed["config.json"])
    changed["config.json"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="requested checkpoint"):
        parity.validate_runtime_checkpoint(runtime, changed, "encoder")
    with pytest.raises(RuntimeError, match="wrong runtime role"):
        parity.validate_runtime_checkpoint(runtime, identity, "decoder")


def _run_exact_transcript(
    tmp_path: Path,
    *,
    encoder_env: str,
    decoder_env: str,
    wav_env: str,
    expected_tokens: list[int],
    expected_text: str,
) -> None:
    encoder = os.environ.get(encoder_env)
    decoder = os.environ.get(decoder_env)
    wav = os.environ.get(wav_env)
    if not all((encoder, decoder, wav)):
        pytest.skip(f"set {encoder_env}, {decoder_env}, and {wav_env}")
    report = tmp_path / f"{encoder_env.lower()}.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--encoder-run-dir",
            encoder,
            "--decoder-run-dir",
            decoder,
            "--wav",
            wav,
            "--language",
            "en",
            "--task",
            "transcribe",
            "--max-tokens",
            "64",
            "--output",
            str(report),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["decoder"]["generated_tokens"] == expected_tokens
    assert result["decoder"]["stop"] == "eos"
    assert result["decoder"]["text"] == expected_text


def test_whisper_tiny_jfk_exact_transcript_when_artifacts_are_configured(
    tmp_path: Path,
) -> None:
    _run_exact_transcript(
        tmp_path,
        encoder_env="CK_WHISPER_ENCODER_RUN_DIR",
        decoder_env="CK_WHISPER_DECODER_RUN_DIR",
        wav_env="CK_WHISPER_WAV",
        expected_tokens=[
            400, 370, 452, 7177, 6280, 1029, 406, 437, 428, 1941, 393, 360,
            337, 291, 1029, 437, 291, 393, 360, 337, 428, 1941, 13,
        ],
        expected_text=(
            " And so my fellow Americans ask not what your country can do for "
            "you ask what you can do for your country."
        ),
    )


def test_whisper_base_jfk_exact_transcript_when_artifacts_are_configured(
    tmp_path: Path,
) -> None:
    _run_exact_transcript(
        tmp_path,
        encoder_env="CK_WHISPER_BASE_ENCODER_RUN_DIR",
        decoder_env="CK_WHISPER_BASE_DECODER_RUN_DIR",
        wav_env="CK_WHISPER_BASE_WAV",
        expected_tokens=[
            400, 370, 452, 7177, 6280, 11, 1029, 406, 437, 428, 1941, 393,
            360, 337, 291, 11, 1029, 437, 291, 393, 360, 337, 428, 1941, 13,
        ],
        expected_text=(
            " And so my fellow Americans, ask not what your country can do for "
            "you, ask what you can do for your country."
        ),
    )
