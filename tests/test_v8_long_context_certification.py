from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_long_context_certification_v8.py"
SPEC = importlib.util.spec_from_file_location("run_long_context_certification_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_catalog_covers_promoted_long_context_families() -> None:
    payload = runner.load_schema(
        ROOT / "version" / "v8" / "regression" / "long_context_models.json",
        "cke.v8.long_context_model_catalog",
    )
    assert {row["id"] for row in payload["models"]} == {
        "qwen36_27b",
        "qwen38_27b",
        "qwen35_35b_a3b",
        "gemma4_e4b",
        "glm4_9b",
        "nemotron_nano_9b",
        "instella_moe_16b_a3b",
        "kimi_vl_a3b",
        "laguna_s_2_1",
        "cohere2_command_r7b",
    }


def test_default_context_ladder_reaches_256k() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'default="2048,8192,32768,65536,131072,262144"' in source
    assert '"capacity_workload": "deterministic_fixed_token"' in source
    assert '"workload": "deterministic_fixed_token"' in source


def test_context_parser_is_sorted_unique_and_fail_closed() -> None:
    assert runner.parse_contexts("8192,2048,8192") == [2048, 8192]
    for value in ("", "0", "2048,-1"):
        try:
            runner.parse_contexts(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid contexts: {value!r}")


def test_fixed_token_hash_is_stable_and_count_sensitive() -> None:
    assert runner.token_sha256(100, 2048) == runner.token_sha256(100, 2048)
    assert runner.token_sha256(100, 2048) != runner.token_sha256(100, 2047)


def test_native_timing_and_peak_rss_are_parsed() -> None:
    timing = runner.parse_timing(
        "prefill 2048 tok (2048 user + 0 tmpl)  1000.0 ms  2048.0 tok/s"
        " | decode 8 tok  80.0 ms  100.0 tok/s  10.0 ms/tok | total 1080.0 ms"
    )
    assert timing["prompt_tokens"] == 2048
    assert timing["decode_tokens"] == 8
    assert runner.parse_peak_rss("Maximum resident set size (kbytes): 123456") == 123456


def test_time_binary_resolution_is_path_driven_and_optional() -> None:
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "time"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        assert runner.resolve_time_binary({"PATH": directory}) == str(executable)
        assert runner.resolve_time_binary({"PATH": ""}) is None
        assert runner.resolve_time_binary({"CK_TIME_BIN": str(executable)}) == str(executable)
        executable.chmod(0o644)
        assert runner.resolve_time_binary({"CK_TIME_BIN": str(executable)}) is None


def test_build_failures_distinguish_capacity_from_contract_faults() -> None:
    assert runner.classify_build_failure({
        "returncode": 137,
        "stderr": "Killed",
    })[0] == "SKIP"
    assert runner.classify_build_failure({
        "returncode": 1,
        "stderr": "HARD CALL ABI FAULT",
    })[0] == "FAIL"


def test_provider_summary_records_selected_kernel_counts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = {
            "operations": [
                {"kernel": "alpha", "op": "a", "layer": 0},
                {
                    "kernel": "beta",
                    "op": "b",
                    "layer": 0,
                    "resolved_contract": {"contract_id": "exact", "selector": "shape"},
                },
                {"kernel": "alpha", "op": "c", "layer": 1},
            ]
        }
        for name in ("lowered_prefill.json", "lowered_decode.json"):
            (root / name).write_text(json.dumps(payload), encoding="utf-8")
        summary = runner.provider_summary(root)
        assert summary["prefill"]["kernel_counts"] == {"alpha": 2, "beta": 1}
        assert summary["decode"]["contract_selections"][0]["contract_id"] == "exact"
        assert summary["decode"]["contract_selections"][0]["count"] == 1


def test_median_timing_uses_all_successful_repetitions() -> None:
    samples = [
        {"timing": {"prompt_tokens": 8, "prompt_ms": 3, "prompt_tok_s": 4,
                    "decode_tokens": 2, "decode_ms": 5, "decode_tok_s": 6}},
        {"timing": {"prompt_tokens": 8, "prompt_ms": 7, "prompt_tok_s": 8,
                    "decode_tokens": 2, "decode_ms": 9, "decode_tok_s": 10}},
    ]
    timing = runner.median_timing(samples)
    assert timing is not None
    assert timing["prompt_ms"] == 5


def test_quality_checks_reject_corruption_and_accept_structured_outputs() -> None:
    code = runner.code_quality(
        "```c\n#include <stdio.h>\n```\n"
        "```python\nprint('python')\n```\n"
        "```sql\nSELECT * FROM readings;\n```\n" + "Explanation. " * 20
    )
    assert code["pass"]
    svg = runner.svg_quality(
        '<svg viewBox="0 0 100 100"><title>CPU</title><desc>Pipeline</desc>'
        '<text>tokenize prefill decode detokenize</text><rect/><rect/><rect/><rect/>'
        '<line/><line/><circle/></svg>'
    )
    assert svg["pass"]


def test_native_cli_has_no_fixed_32k_context_clamp_and_traces_logits() -> None:
    source = (ROOT / "version" / "v8" / "src" / "ck_cli_v8.c").read_text(encoding="utf-8")
    assert "CK_CLI_MAX_CONTEXT" not in source
    assert "first_logits_fnv1a64" in source
    assert "explicit prompt has %d tokens" in source
