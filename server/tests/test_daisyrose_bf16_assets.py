"""Run the pinned DaisyRose acceptance against a configured native endpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_served_daisyrose_bf16(tmp_path):
    configuration = os.environ.get("CK_DAISYROSE_BF16_ACCEPTANCE")
    if not configuration:
        pytest.skip("CK_DAISYROSE_BF16_ACCEPTANCE is not configured")
    options = json.loads(Path(configuration).read_text())
    required = {"bundle", "checkpoint", "endpoint", "session-library", "max-abs", "max-rmse"}
    if not required <= options.keys() or options.keys() - (required | {"model", "oracle-threads"}):
        pytest.fail("invalid DaisyRose acceptance configuration")
    output = tmp_path / "evidence"
    command = [sys.executable, "-m", "server.daisyrose_bf16_acceptance",
               "--output-dir", str(output)]
    for key, value in options.items():
        command.extend((f"--{key}", str(value)))
    result = subprocess.run(command, check=False)
    report = json.loads((output / "summary.json").read_text())
    assert result.returncode == 0 and report["status"] == "pass", report
