from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'version/v8/scripts/certify_visual_session_v8.py'


def test_missing_assets_replace_previous_pass(tmp_path):
    report=tmp_path/'latest.json';report.write_text('{"status":"PASS"}')
    result=subprocess.run([sys.executable,str(SCRIPT),'--bridge-report',str(tmp_path/'missing.json'),
        '--prefix-f32',str(tmp_path/'missing.f32'),'--reference-logits-f32',str(tmp_path/'logits.f32'),
        '--output',str(report)],capture_output=True,text=True,timeout=20)
    assert result.returncode==1
    assert json.loads(report.read_text())['status']=='FAIL'


def test_loaded_artifact_mismatch_detected_at_identity_gate(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec=importlib.util.spec_from_file_location('visual_probe_test',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    requested=tmp_path/'requested.so';requested.write_bytes(b'current')
    stale=tmp_path/'stale.so';stale.write_bytes(b'old')
    monkeypatch.setattr(module,'_resolved_symbol_library',lambda *args:stale)
    with pytest.raises(RuntimeError,match='differs from requested'):
        module.checked_loaded_identity(None,'test_symbol',requested)
    stale.write_bytes(requested.read_bytes())
    assert module.checked_loaded_identity(None,'test_symbol',requested)['sha256']==module.sha_file(requested)
