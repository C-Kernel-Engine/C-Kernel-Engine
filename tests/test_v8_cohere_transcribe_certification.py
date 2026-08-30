#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/certify_cohere_transcribe_oracle_v8.py"
SCHEMA = ROOT / "version/v8/schemas/checkpoint_manifest.schema.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("cohere_transcribe_certification", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


certification = _load_module()


def _fake_oracle(path: Path, *, missing_last_block: bool = False) -> None:
    last = 47 if missing_last_block else 48
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import struct
import sys
from pathlib import Path

if '--version' in sys.argv:
    print('fake-crispasr 1.0')
    raise SystemExit(0)

def dump(path, shape):
    count = 1
    for extent in shape:
        count *= extent
    Path(path).write_bytes(struct.pack('<' + 'i' * len(shape), *shape) + struct.pack('<' + 'f' * count, *range(count)))

stages = Path(os.environ['CRISPASR_COHERE_DUMP_STAGES'])
stages.mkdir(parents=True, exist_ok=True)
dump(stages / 'crisp.mel.bin', (4, 3))
for layer in range({last}):
    dump(stages / f'crisp.block{{layer}}.bin', (8, 2))
dump(stages / 'crisp.enc_final.bin', (8, 2))
dump(Path(os.environ['CRISPASR_COHERE_DUMP_ENCOUT']), (6, 2))
if 'CRISPASR_COHERE_DUMP_ATTN' in os.environ:
    dump(Path(os.environ['CRISPASR_COHERE_DUMP_ATTN']), (2, 3, 4))
out = Path(sys.argv[sys.argv.index('-of') + 1]).with_suffix('.json')
out.write_text(json.dumps({{'transcription': 'Ask not what your country can do for you.'}}))
print('cohere:  enc compute       12.5 ms')
print('cohere:  dec compute        7.5 ms')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run(tmp_path: Path, *, missing_last_block: bool = False) -> subprocess.CompletedProcess[str]:
    oracle = tmp_path / "crispasr"
    model = tmp_path / "model.gguf"
    audio = tmp_path / "audio.wav"
    _fake_oracle(oracle, missing_last_block=missing_last_block)
    model.write_bytes(b"model")
    audio.write_bytes(b"audio")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--oracle-bin",
            str(oracle),
            "--model",
            str(model),
            "--audio",
            str(audio),
            "--output-dir",
            str(tmp_path / "report"),
            "--threads",
            "4",
            "--dump-attention",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_oracle_capture_publishes_reproducible_xray_inputs(tmp_path: Path) -> None:
    completed = _run(tmp_path)
    assert completed.returncode == 0, completed.stdout
    report = json.loads((tmp_path / "report/summary.json").read_text())
    manifest = json.loads(
        (tmp_path / "report/oracle_checkpoint_manifest.json").read_text()
    )
    schema = json.loads(SCHEMA.read_text())
    assert not list(Draft202012Validator(schema).iter_errors(manifest))
    assert report["status"] == "PASS"
    assert report["checkpoint_count"] == 52
    assert report["threads"] == 4
    assert report["timings_ms"]["enc_compute"] == 12.5
    assert len(report["oracle"]["sha256"]) == 64
    assert len(report["model"]["sha256"]) == 64
    assert manifest["checkpoints"][0]["logical_shape"] == [3, 4]
    assert manifest["checkpoints"][-1]["logical_shape"] == [2, 3, 4]
    assert manifest["checkpoints"][1]["checkpoint_id"] == "audio.encoder.layer.0.output"
    assert manifest["checkpoints"][48]["checkpoint_id"] == "audio.encoder.layer.47.output"


def test_oracle_capture_fails_closed_when_a_conformer_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "report/oracle_raw/crisp.block47.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale checkpoint must not survive a rerun")
    completed = _run(tmp_path, missing_last_block=True)
    assert completed.returncode == 2
    report = json.loads((tmp_path / "report/summary.json").read_text())
    assert report["status"] == "FAIL"
    assert "all 48 Conformer blocks" in report["reason"]
    assert "missing=[47]" in report["reason"]


def test_headered_tensor_reader_rejects_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(struct.pack("<2i", 4, 3) + b"short")
    with pytest.raises(certification.CertificationError, match="expected 48"):
        certification._read_headered_f32(path, 2)


def test_artifact_backed_make_target_is_fail_closed_and_portable() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "test-cohere-transcribe-oracle-auto:" in makefile
    assert "CK_COHERE_TRANSCRIBE_ORACLE" in makefile
    assert "CK_COHERE_TRANSCRIBE_MODEL" in makefile
    assert "CK_COHERE_TRANSCRIBE_AUDIO" in makefile
    assert "/data/" not in makefile[
        makefile.index("test-cohere-transcribe-oracle-auto:") :
        makefile.index("# Policy:")
    ]
