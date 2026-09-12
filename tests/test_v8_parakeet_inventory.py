import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "version/v8/contracts/parakeet_tdt_0_6b_v3_inventory.json"
FIXTURE_JSON = ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_2086-149220-0033_fp32.json"
FIXTURE_NPZ = ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_2086-149220-0033_fp32.npz"
TENSOR_MANIFEST = ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_tensor_manifest.json"
AUDIO_DIR = ROOT / "version/v8/test_assets/parakeet_tdt_0_6b_v3"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parakeet_operation_inventory_is_complete_and_honest():
    inventory = _json(INVENTORY)
    operations = inventory["operations"]
    counts = Counter(item["disposition"] for item in operations)

    assert inventory["status"] == "native_five_minute_chunked_e2e_pass"
    assert inventory["summary"] == {"operations": len(operations), **dict(counts)}
    assert len(operations) == 30
    assert len({item["operation"] for item in operations}) == len(operations)
    assert all(item["finding"] and item["cke_foundation"] for item in operations)
    assert inventory["claim_boundary"] == {
        "native_cke_transcription": "five_minute_full_and_chunked_e2e_pass",
        "native_cke_parity": "exact_50_token_and_duration_trajectory",
        "long_audio": "five_minute_deterministic_overlap_pass_42_minute_not_tested",
        "multilingual": "not_tested",
        "diarization": "out_of_scope",
        "bump_conversion": "699_mapped_24_explicitly_ignored_0_unaccounted",
        "resampling": "synthetic_48khz_stereo_short_fixture_pass",
    }
    by_operation = {item["operation"]: item for item in operations}
    assert (
        by_operation["inference BatchNorm1D"]["cke_foundation"]
        == "audio_batch_norm_inference_channel_major_f32"
    )
    assert (
        by_operation["inference BatchNorm1D"]["implementation_status"]
        == "short_fixture_validated"
    )
    lstm = by_operation["two-layer 640-wide LSTM prediction network"]
    assert lstm["cke_foundation"] == "audio_lstm_step_f32"
    assert lstm["implementation_status"] == "short_fixture_validated"
    assert {
        item["operation"]
        for item in operations
        if item["implementation_status"] == "open"
    } == {
        "PCM decode and mono conversion",
        "sample-rate validation and resampling",
        "token timestamps",
    }


def test_parakeet_reference_fixture_arrays_are_present_finite_and_hashed():
    inventory = _json(INVENTORY)
    report = _json(FIXTURE_JSON)
    assert report["status"] == "pass"
    assert report["checks"] == {
        "required_arrays_present": True,
        "missing_arrays": [],
        "all_arrays_finite": True,
        "nonempty_trajectory": True,
    }
    assert report["arrays_file"]["sha256"] == _sha256(FIXTURE_NPZ)
    assert report["decode"]["steps"] > 1
    assert report["decode"]["nonblank_steps"] > 0
    assert report["decode"]["transcript"]
    assert report["decode"]["timestamps"]
    assert report["model"]["revision"] == inventory["reference_contract"]["model"]["revision"]
    assert report["reference"]["revision"] == inventory["reference_contract"]["implementation"]["revision"]
    assert report["reference"]["source_sha256"] == inventory["reference_contract"]["implementation"]["source_sha256"]

    with np.load(FIXTURE_NPZ, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(report["arrays_file"]["arrays"])
        for name, expected in report["arrays_file"]["arrays"].items():
            value = np.ascontiguousarray(arrays[name])
            assert list(value.shape) == expected["shape"]
            assert str(value.dtype) == expected["dtype"]
            assert np.isfinite(value).all()
            assert hashlib.sha256(value.tobytes(order="C")).hexdigest() == expected["sha256"]


def test_parakeet_audio_and_tensor_manifest_provenance():
    source = _json(AUDIO_DIR / "source.json")
    audio = AUDIO_DIR / source["asset"]
    report = _json(FIXTURE_JSON)
    manifest = _json(TENSOR_MANIFEST)

    assert source["corpus_license"] == "CC BY 4.0"
    assert audio.stat().st_size == source["bytes"]
    assert _sha256(audio) == source["sha256"] == report["audio"]["sha256"]
    assert manifest["complete"] is True
    assert manifest["tensor_count"] == len(manifest["tensors"]) == 723
    assert len({tensor["name"] for tensor in manifest["tensors"]}) == 723
    assert manifest["tensor_bytes"] == sum(tensor["bytes"] for tensor in manifest["tensors"])
    assert manifest["model"]["weights_sha256"] == report["model"]["weights"]["sha256"]
    names = {tensor["name"] for tensor in manifest["tensors"]}
    assert {"decoder.embedding.weight", "decoder.lstm.weight_ih_l0", "joint.head.weight"} <= names
    assert "encoder.layers.23.self_attn.relative_k_proj.weight" in names
