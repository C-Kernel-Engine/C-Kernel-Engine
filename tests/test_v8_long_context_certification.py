from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


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
        "qwen36_35b_a3b",
        "qwen38_27b",
        "qwen35_35b_a3b",
        "gemma4_e4b",
        "glm4_9b",
        "nemotron_nano_9b",
        "instella_moe_16b_a3b",
        "kimi_vl_a3b",
        "laguna_s_2_1",
        "laguna_xs_2_1",
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
    usage = runner.parse_resource_usage(
        "User time (seconds): 120.0\nSystem time (seconds): 8.0\n"
        "Percent of CPU this job got: 1600%\n",
        8.0,
    )
    assert usage is not None
    assert usage["average_cpu_cores"] == 16.0
    assert usage["reported_cpu_percent"] == 1600.0


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


def test_resource_usage_falls_back_to_child_cpu_deltas() -> None:
    before = SimpleNamespace(ru_utime=10.0, ru_stime=2.0)
    after = SimpleNamespace(ru_utime=22.0, ru_stime=4.0)
    usage = runner.resource_usage_delta(before, after, 2.0)
    assert usage["process_seconds"] == 14.0
    assert usage["average_cpu_cores"] == 7.0
    assert usage["reported_cpu_percent"] == 700.0


def test_cpu_plan_selects_one_allowed_sibling_per_physical_core() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        topology = {
            0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (0, 1),
        }
        for cpu, (package, core) in topology.items():
            path = root / f"cpu{cpu}" / "topology"
            path.mkdir(parents=True)
            (path / "physical_package_id").write_text(str(package), encoding="ascii")
            (path / "core_id").write_text(str(core), encoding="ascii")
        plan = runner.resolve_cpu_plan(
            0, "physical", allowed_cpus={0, 1, 2, 3}, sysfs_root=root
        )
        assert plan["effective_threads"] == 2
        assert plan["physical_representatives"] == [0, 1]
        assert plan["affinity"] == [0, 1]


def test_utilization_summary_exposes_low_occupancy_intervals() -> None:
    samples = [
        {"elapsed_seconds": 0.0, "active_cores": None, "rss_kib": 10},
        {"elapsed_seconds": 0.25, "active_cores": 0.5, "rss_kib": 20,
         "per_cpu_busy_pct": {"0": 20.0, "1": 40.0}},
        {"elapsed_seconds": 0.50, "active_cores": 0.5, "rss_kib": 30,
         "per_cpu_busy_pct": {"0": 100.0, "1": 80.0}},
        {"elapsed_seconds": 0.75, "active_cores": 7.5, "rss_kib": 25,
         "per_cpu_busy_pct": {"0": 100.0, "1": 100.0}},
        {"elapsed_seconds": 1.00, "active_cores": 8.0, "rss_kib": 20,
         "per_cpu_busy_pct": {"0": 100.0, "1": 100.0}},
    ]
    summary = runner.summarize_utilization_samples(samples, requested_threads=8)
    assert summary["sample_count"] == 4
    assert summary["peak_sampled_rss_kib"] == 30
    assert summary["longest_low_utilization_seconds"] == 0.75
    assert summary["p90_active_cores"] > 7.0
    assert summary["mean_selected_cpu_active_cores"] == 1.6
    assert summary["minimum_per_cpu_mean_utilization_pct"] == 80.0
    assert summary["maximum_per_cpu_mean_utilization_pct"] == 80.0
    assert summary["all_sampled_cpus_ge_80pct_fraction"] == 0.75


def test_top_like_per_cpu_parser_tracks_busy_and_total_ticks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        proc_stat = Path(directory) / "stat"
        proc_stat.write_text(
            "cpu  20 0 10 70 0 0 0 0 0 0\n"
            "cpu0 10 0 5 35 0 0 0 0 0 0\n"
            "cpu1 10 0 5 30 5 0 0 0 0 0\n",
            encoding="ascii",
        )
        assert runner._system_cpu_totals({1}, proc_stat) == {1: (15, 50)}


def test_timed_command_publishes_process_utilization_timeline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        env = os.environ.copy()
        env["CK_NUM_THREADS"] = "1"
        result = runner.run_command(
            [
                sys.executable,
                "-c",
                "import time\nend=time.monotonic()+0.6\nwhile time.monotonic()<end: pass",
            ],
            timeout=5,
            env=env,
            output_dir=output,
            name="busy",
            timed=True,
        )
        assert result["returncode"] == 0
        assert result["utilization"]["sample_count"] >= 1
        timeline = json.loads(
            Path(result["utilization_timeline_path"]).read_text(encoding="utf-8")
        )
        assert timeline["schema"] == "cke.process_utilization_timeline"
        assert timeline["summary"]["p50_active_cores"] > 0.5


def test_build_failures_distinguish_capacity_from_contract_faults() -> None:
    assert runner.classify_build_failure({
        "returncode": 137,
        "stderr": "Killed",
    })[0] == "SKIP"
    assert runner.classify_build_failure({
        "returncode": 1,
        "stderr": "HARD CALL ABI FAULT",
    })[0] == "FAIL"


def test_quality_context_plan_allocates_exact_total_and_remaining_decode() -> None:
    prompts = [{"max_tokens": 6144}, {"max_tokens": 8192}]
    fixed = runner.quality_context_plan(8192, prompts, 131072)
    assert fixed == {
        "mode": "fixed_total_context",
        "total_context_tokens": 131072,
        "input_reserve_tokens": 8192,
        "output_budget_tokens": 122872,
    }
    legacy = runner.quality_context_plan(8192, prompts, 0)
    assert legacy["total_context_tokens"] == 16392
    assert legacy["output_budget_tokens"] is None
    try:
        runner.quality_context_plan(8192, prompts, 8200)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted total context without decode capacity")


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


def test_engineering_quality_warns_for_coherent_imperfect_artifacts() -> None:
    coherent_but_incomplete = (
        "Here is a coherent implementation plan with a complete explanation of the "
        "input, output, vector loop, scalar tail, and numerical ordering constraints. "
        "The requested source is intentionally omitted from this fixture."
    )
    code = runner.code_quality(coherent_but_incomplete)
    assert code["coherence_pass"]
    assert not code["pass"]
    assert runner.engineering_quality_status(0, code, True, True) == "WARN"

    malformed_svg = (
        "The following diagram explains the vector lanes, scalar tail, memory flow, "
        "and numerical ordering in a coherent form for human review. "
        "<svg><title>RoPE pipeline</title><text>lane flow and scalar tail</svg>"
    )
    svg = runner.svg_quality(malformed_svg)
    assert svg["coherence_pass"]
    assert not svg["pass"]
    assert runner.engineering_quality_status(0, svg, True, True) == "WARN"


def test_engineering_quality_fails_corrupted_model_output() -> None:
    corrupted = "g\ufffd\\uFFFD\x03" * 80
    quality = runner.code_quality(corrupted)
    assert not quality["coherence_pass"]
    assert runner.engineering_quality_status(0, quality, True, True) == "FAIL"


def test_engineering_quality_contract_is_paired_and_fail_closed() -> None:
    payload = runner.load_schema(
        ROOT / "version" / "v8" / "test_assets" / "long_context_quality_prompts.json",
        "cke.v8.long_context_quality_prompts",
    )
    prompts = runner.validate_quality_prompts(payload)
    assert [row["id"] for row in prompts] == ["rope_simd_kernel", "rope_simd_svg"]
    assert prompts[1]["depends_on"] == prompts[0]["id"]
    materialized = runner.materialize_quality_prompt(
        prompts[1], {prompts[0]["id"]: "int generated_kernel(void);\n"}
    )
    assert "int generated_kernel(void);" in materialized
    assert "{{dependency_output}}" not in materialized
    prefixed = runner.materialize_quality_prompt(prompts[0], {}, "dossier body\n")
    assert prefixed.startswith("dossier body\n\nTask:\n")
    dependent = runner.materialize_quality_prompt(
        prompts[1], {prompts[0]["id"]: "int generated_kernel(void);\n"}, "ignored"
    )
    assert "ignored" not in dependent

    broken = json.loads(json.dumps(payload))
    broken["prompts"][0]["depends_on"] = "later"
    try:
        runner.validate_quality_prompts(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a forward or missing quality dependency")


def test_generated_response_extraction_removes_cli_framing() -> None:
    stdout = (
        "Prompt: task\nResponse: ```c\n#include <stddef.h>\nint main(void) { return 0; }\n```\n"
        "prefill 12 tok  1.0 ms  12.0 tok/s | decode 2 tok  1.0 ms  2.0 tok/s\n"
        "goodbye\n"
    )
    generated = runner.extract_generated(stdout)
    assert generated.startswith("```c")
    assert "Prompt:" not in generated
    assert "prefill" not in generated
    assert "goodbye" not in generated.lower()
    assert runner.extract_c_source(generated).startswith("#include <stddef.h>")


def test_generated_c_is_strictly_compiled_but_never_executed() -> None:
    source = """```c
#include <immintrin.h>
#include <stddef.h>

static void ck_example(float *dst, const float *src) {
    __m256 value = _mm256_loadu_ps(src);
    _mm256_storeu_ps(dst, value);
}

int main(void) {
    float src[8] = {0.0f};
    float dst[8] = {0.0f};
    ck_example(dst, src);
    return dst[0] != 0.0f;
}
```"""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        artifact = output / "kernel.c"
        quality = runner.code_quality(
            source,
            artifact,
            {
                "kind": "c_kernel.v1",
                "required_fragments": ["#include <immintrin.h>", "_mm256_loadu_ps"],
                "forbidden_fragments": ["malloc("],
            },
            os.environ.copy(),
            output,
        )
        assert quality["pass"]
        assert quality["strict_compile"] is True
        assert quality["compile_scope"].startswith("syntax-only")
        assert artifact.read_text(encoding="utf-8").startswith("#include <immintrin.h>")


def test_quality_runner_feeds_clean_c_into_svg_and_publishes_gallery(monkeypatch) -> None:
    code = """```c
#include <immintrin.h>
#include <stddef.h>
static void ck_pair(float *dst, const float *src) {
    __m256 value = _mm256_loadu_ps(src);
    _mm256_storeu_ps(dst, value);
}
int main(void) {
    float src[8] = {0.0f};
    float dst[8] = {0.0f};
    ck_pair(dst, src);
    return dst[0] != 0.0f;
}
```"""
    svg = (
        '<svg viewBox="0 0 100 100"><title>RoPE</title><desc>AVX2</desc>'
        '<defs><marker id="arrow"><path d="M0 0L4 2L0 4z"/></marker></defs>'
        '<text>ck_pair lanes tail no allocation</text>'
        '<rect/><rect/><rect/><rect/><rect/><rect/><line marker-end="url(#arrow)"/></svg>'
    )
    real_run_command = runner.run_command
    observed_prompts: list[str] = []

    def fake_run_command(command, **kwargs):
        if "-fsyntax-only" in command:
            return real_run_command(command, **kwargs)
        prompt_text = command[command.index("--prompt") + 1]
        observed_prompts.append(prompt_text)
        generated = code if len(observed_prompts) == 1 else svg
        return {
            "returncode": 0,
            "stdout": (
                f"Response: {generated}\n"
                "prefill 32 tok  10.0 ms  3200.0 tok/s | "
                "decode 4 tok  2.0 ms  2000.0 tok/s\n"
            ),
            "stderr": "",
            "peak_rss_kib": 100,
            "resource_usage": {"average_cpu_cores": 8.0},
            "stdout_path": str(output / "model.stdout.log"),
            "stderr_path": str(output / "model.stderr.log"),
        }

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    prompts = [
        {
            "id": "code", "kind": "c_kernel.v1", "artifact_extension": ".c",
            "max_tokens": 64, "text": "write code",
            "required_fragments": ["#include <immintrin.h>", "_mm256_loadu_ps"],
            "forbidden_fragments": ["malloc("],
        },
        {
            "id": "svg", "kind": "kernel_svg.v1", "artifact_extension": ".svg",
            "depends_on": "code", "max_tokens": 64,
            "min_graphic_elements": 8,
            "required_labels": ["ck_pair", "lanes", "tail", "no allocation"],
            "text": "explain this source: {{dependency_output}}",
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        rows = runner.run_quality(
            {"id": "test", "label": "Test Model"},
            output / "runtime", 256, prompts,
            SimpleNamespace(resume=False, ck_cli=output / "ck-cli", quality_timeout=10),
            os.environ.copy(), output / "quality" / "test",
        )
        assert [row["status"] for row in rows] == ["PASS", "PASS"]
        assert "#include <immintrin.h>" in observed_prompts[1]
        assert Path(rows[0]["artifact_path"]).read_text(encoding="utf-8").startswith(
            "#include <immintrin.h>"
        )
        report = {"quality": rows}
        runner.write_quality_index(report, output)
        gallery = (output / "quality" / "index.html").read_text(encoding="utf-8")
        assert "View generated C" in gallery
        assert "Generated SIMD kernel diagram" in gallery


def test_quality_runner_enforces_root_prefill_and_invalidates_stale_resume(
    monkeypatch,
) -> None:
    generated = "```c\n#include <stddef.h>\nint main(void) { return 0; }\n```\n" + "x" * 300
    inference_calls = 0

    def fake_run_command(command, **kwargs):
        nonlocal inference_calls
        if "--prompt" in command:
            inference_calls += 1
        return {
            "returncode": 0,
            "stdout": (
                f"Response: {generated}\n"
                "prefill 31 tok  10.0 ms  3100.0 tok/s | "
                "decode 4 tok  2.0 ms  2000.0 tok/s\n"
            ),
            "stderr": "",
            "peak_rss_kib": 100,
            "resource_usage": {"average_cpu_cores": 8.0},
            "stdout_path": str(kwargs["output_dir"] / "model.stdout.log"),
            "stderr_path": str(kwargs["output_dir"] / "model.stderr.log"),
        }

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    prompt = {
        "id": "code", "kind": "c_kernel.v1", "artifact_extension": ".c",
        "max_tokens": 64, "text": "write code",
    }
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        args = SimpleNamespace(
            resume=True, ck_cli=output / "ck-cli", quality_timeout=10,
            quality_prefix="dossier", quality_min_root_prefill_tokens=32,
        )
        first = runner.run_quality(
            {"id": "test", "label": "Test Model"},
            output / "runtime", 256, [prompt], args,
            os.environ.copy(), output / "quality" / "test",
        )
        assert first[0]["status"] == "FAIL"
        assert first[0]["prefill_consumption_verified"] is False
        args.quality_prefix = "changed dossier"
        runner.run_quality(
            {"id": "test", "label": "Test Model"},
            output / "runtime", 256, [prompt], args,
            os.environ.copy(), output / "quality" / "test",
        )
        assert inference_calls == 2


def test_quality_runner_uses_remaining_total_context_until_native_eos(
    monkeypatch,
) -> None:
    generated = "```c\n#include <stddef.h>\nint main(void) { return 0; }\n```\n" + "x" * 300
    inference_commands: list[list[str]] = []

    def fake_run_command(command, **kwargs):
        if "--prompt" in command:
            inference_commands.append(command)
            trace_path = Path(command[command.index("--token-trace-json") + 1])
            trace_path.write_text(
                json.dumps({"prompt_tokens": 100, "generated_tokens": 200,
                            "stop_reason": "eos"}),
                encoding="utf-8",
            )
        return {
            "returncode": 0,
            "stdout": (
                f"Response: {generated}\n"
                "prefill 100 tok  10.0 ms  10000.0 tok/s | "
                "decode 200 tok  20.0 ms  10000.0 tok/s\n"
            ),
            "stderr": "",
            "peak_rss_kib": 100,
            "resource_usage": {"average_cpu_cores": 8.0},
            "stdout_path": str(kwargs["output_dir"] / "model.stdout.log"),
            "stderr_path": str(kwargs["output_dir"] / "model.stderr.log"),
        }

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    prompt = {
        "id": "code", "kind": "c_kernel.v1", "artifact_extension": ".c",
        "max_tokens": 64, "text": "write code",
    }
    context_plan = runner.quality_context_plan(8192, [prompt], 131072)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        rows = runner.run_quality(
            {"id": "test", "label": "Test Model"},
            output / "runtime", 131072, [prompt],
            SimpleNamespace(
                resume=False, ck_cli=output / "ck-cli", quality_timeout=10,
                quality_prefix="", quality_min_root_prefill_tokens=0,
            ),
            os.environ.copy(), output / "quality" / "test",
            context_plan=context_plan,
        )
        command = inference_commands[0]
        assert command[command.index("--context") + 1] == "131072"
        assert command[command.index("--max-tokens") + 1] == "122872"
        assert rows[0]["output_budget_tokens"] == 122872
        assert rows[0]["native_stop_reason"] == "eos"
        assert rows[0]["configured_prompt_cap_tokens"] == 64


def test_native_cli_has_no_fixed_32k_context_clamp_and_traces_logits() -> None:
    source = (ROOT / "version" / "v8" / "src" / "ck_cli_v8.c").read_text(encoding="utf-8")
    assert "CK_CLI_MAX_CONTEXT" not in source
    assert "first_logits_fnv1a64" in source
    assert "explicit prompt has %d tokens" in source
