import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRINT_TARGET = """\
.PHONY: print-dnnl-selection
print-dnnl-selection:
\t@printf 'DNNL_INC=<%s> DNNL_LIB=<%s>\\n' '$(DNNL_INC)' '$(DNNL_LIB)'
"""


def _make_selection(*assignments: str, environment: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if environment:
        env.update(environment)
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-s",
            "-f",
            "Makefile",
            "-f",
            "-",
            "USE_ONEDNN=1",
            *assignments,
            "print-dnnl-selection",
        ],
        cwd=ROOT,
        env=env,
        input=PRINT_TARGET,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def test_onednn_environment_paths_override_auto_detection() -> None:
    output = _make_selection(
        environment={
            "DNNL_INC": "/opt/oracle/include -I/opt/oracle/source/include",
            "DNNL_LIB": "/opt/oracle/lib",
        }
    )
    assert (
        "DNNL_INC=</opt/oracle/include -I/opt/oracle/source/include> "
        "DNNL_LIB=</opt/oracle/lib>"
    ) in output


def test_onednn_command_line_root_overrides_auto_detection() -> None:
    output = _make_selection("DNNL_ROOT=/opt/onednn-exact")
    assert (
        "DNNL_INC=</opt/onednn-exact/include> "
        "DNNL_LIB=</opt/onednn-exact/lib>"
    ) in output
