import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAB = load("bench_v8_family_lab", ROOT / "benchmarks" / "bench_v8_family_lab.py")
COMPARE = load("compare_ck_llama_v8", ROOT / "benchmarks" / "compare_ck_llama_v8.py")


def test_external_manifests_are_valid() -> None:
    prompts = COMPARE.load_prompts(ROOT / "benchmarks" / "fixtures" / "v8_lab_prompts.json")
    models = COMPARE.load_models(ROOT / "benchmarks" / "fixtures" / "v8_lab_models.json")
    assert "email_summary" in prompts
    assert "svg_infographic" in prompts
    assert len(models) >= 6
    assert len({model.key for model in models}) == len(models)


def test_quality_checks_are_structural_not_text_identity() -> None:
    json_features = LAB.quality_features("structured_json", '{"status":"ok"}')
    svg_features = LAB.quality_features(
        "svg_infographic",
        '<svg viewBox="0 0 10 10"><title>T</title><desc>D</desc></svg>',
    )
    assert json_features["valid_json"] is True
    assert svg_features["has_svg"] is True
    assert svg_features["has_viewbox"] is True
    assert svg_features["has_accessible_title"] is True


def test_slowest_case_is_selected_per_model() -> None:
    rows = [
        {"model_key": "a", "model": "A", "prompt_tokens": 32, "ratios": {"prompt": 0.9}},
        {"model_key": "a", "model": "A", "prompt_tokens": 512, "ratios": {"prompt": 0.6}},
        {"model_key": "b", "model": "B", "prompt_tokens": 128, "ratios": {"prompt": 1.1}},
    ]
    result = LAB.slowest_cases(rows)
    by_model = {row["model_key"]: row for row in result}
    assert by_model["a"]["prompt_tokens"] == 512
    assert by_model["b"]["prompt_tokens"] == 128


def test_html_report_escapes_model_output(tmp_path: Path) -> None:
    report = {
        "generated_at": "now",
        "host": {"cpu": "test"},
        "config": {"threads": 1},
        "performance": [],
        "slowest_cases": [],
        "profiler_plan": [],
        "operation_profiles": [],
        "prompt_comparisons": [
            {
                "model": "M",
                "prompt_key": "hello",
                "cke": {"generated": "<script>alert(1)</script>"},
                "llama": {"generated": "ok"},
                "cke_quality": {},
                "llama_quality": {},
                "text_similarity": 0.0,
            }
        ],
    }
    output = tmp_path / "index.html"
    LAB.render_html(report, output)
    rendered = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered


def test_html_report_surfaces_consumption_and_profile_evidence(tmp_path: Path) -> None:
    report = {
        "generated_at": "now",
        "host": {"cpu": "test"},
        "config": {"threads": 1},
        "performance": [{
            "model": "M", "quant": "Q4", "prompt_tokens": 128,
            "cke": {"prompt_ms": 2, "decode_tok_s": 3},
            "llama": {"prompt_ms": 3, "decode_tok_s": 4},
            "ratios": {"prompt": 1.5},
            "consumption": {
                "cke_count_verified": True, "llama_count_verified": True,
                "requested_token_sha256": "abcdef0123456789",
            },
        }],
        "slowest_cases": [], "profiler_plan": [], "prompt_comparisons": [],
        "model_artifacts": [],
        "operation_profiles": [{
            "model_key": "m", "prompt_tokens": 128, "prompt_ms": 2,
            "profiled_ms": 1, "coverage_pct": 50, "top_operations": [],
            "core_equivalents": 0.75, "worker_utilization_pct": 75.0,
            "selected_kernels": [{"kernel": "q5_fast"}],
            "decode_ms": 3, "decode_profiled_ms": 2,
            "decode_coverage_pct": 66.7, "decode_core_equivalents": 0.5,
            "decode_worker_utilization_pct": 50.0,
            "decode_top_operations": [{
                "op": "decode_gemv", "time_ms": 2, "pct": 100,
            }],
            "decode_selected_kernels": [{"kernel": "decode_q5_fast"}],
            "threadpool": {"completion_wait_ms": 0.25},
        }],
        "system_profile_counters": [{
            "model_key": "m", "prompt_tokens": 128,
            "cke": {"cycles": 10}, "llama": {"cycles": 8},
        }],
    }
    output = tmp_path / "index.html"
    LAB.render_html(report, output)
    rendered = output.read_text(encoding="utf-8")
    assert "abcdef012345" in rendered
    assert "q5_fast" in rendered
    assert "decode_q5_fast" in rendered
    assert "decode_gemv" in rendered
    assert "Worker utilization" in rendered
    assert "0.25" in rendered
    assert "Matched hardware counters" in rendered


def test_median_metrics_uses_independent_samples() -> None:
    result = COMPARE.median_metrics(
        [
            {"prompt_ms": 3.0, "prompt_tok_s": 10.0},
            {"prompt_ms": 1.0, "prompt_tok_s": 30.0},
            {"prompt_ms": 2.0, "prompt_tok_s": 20.0},
        ]
    )
    assert result == {"prompt_ms": 2.0, "prompt_tok_s": 20.0}


def test_fixed_token_hash_covers_order_and_count() -> None:
    assert COMPARE.token_sequence_sha256(100, 128) == COMPARE.token_sequence_sha256(100, 128)
    assert COMPARE.token_sequence_sha256(100, 128) != COMPARE.token_sequence_sha256(100, 127)
    assert COMPARE.token_sequence_sha256(100, 128) != COMPARE.token_sequence_sha256(101, 128)


def test_llama_completion_output_excludes_profiler_log() -> None:
    combined = (
        "Generated answer.\n\nwarning: no usable GPU found\n"
        "0.00 I common_perf_print: prompt eval time = 10 ms"
    )
    assert COMPARE.extract_llama_generated(combined) == "Generated answer."


def test_prompt_words_do_not_trigger_ck_timing_parser() -> None:
    output = "Explain how prefill differs from decode in one paragraph."
    assert COMPARE.parse_ck_timing_optional(output) is None


def test_profiler_plan_contains_matched_native_commands(tmp_path: Path) -> None:
    cases = [{
        "model_key": "m", "model": "Model", "prompt_tokens": 32, "context": 44,
        "ratios": {"prompt": 0.5},
    }]
    models = [{
        "key": "m", "gguf": "/models/m.gguf", "ck_run_dir": "/models/m",
        "profile_model_id": "m-profile",
    }]
    plan = LAB.profiler_plan(cases, models, tmp_path, 8, Path("/bin/cke"), Path("/llama"))
    assert plan[0]["cke_command"][0] == "/bin/cke"
    assert "32" in plan[0]["llama_command"]
    names = {row["name"] for row in plan[0]["system_profiles"]}
    assert {"cke_perf_stat", "llama_perf_stat", "cke_vtune_hotspots", "llama_vtune_hotspots"} <= names
    assert len(names) == len(plan[0]["system_profiles"])
    cke_perf = next(row for row in plan[0]["system_profiles"] if row["name"] == "cke_perf_stat")
    assert "CK_THREADPOOL_PROFILE=1" in cke_perf["command"]


def test_slowest_cases_are_ranked_for_profile_limit() -> None:
    rows = [
        {"model_key": "near", "model": "Near", "prompt_tokens": 128, "ratios": {"prompt": 0.9}},
        {"model_key": "far", "model": "Far", "prompt_tokens": 128, "ratios": {"prompt": 0.3}},
        {"model_key": "middle", "model": "Middle", "prompt_tokens": 128, "ratios": {"prompt": 0.6}},
    ]
    assert [row["model_key"] for row in LAB.slowest_cases(rows)[:2]] == ["far", "middle"]


def test_operation_profiles_are_collected_from_artifacts(tmp_path: Path) -> None:
    directory = tmp_path / "profiles" / "m" / "p32"
    directory.mkdir(parents=True)
    (directory / "cke_ops.json").write_text(json.dumps({
        "results": [{
            "run": {"prompt_ms": 20.0, "decode_ms": 8.0},
            "summary": {
                "prefill_total_ms": 15.0,
                "by_op": [{"op": "gemm", "time_ms": 10.0, "pct": 66.7}],
                "by_kernel_op": [{"kernel": "gemm_fast", "op": "gemm", "time_ms": 10.0}],
                "phases": {
                    "decode": {
                        "total_ms": 6.0,
                        "core_equivalents": 3.0,
                        "worker_utilization_pct": 75.0,
                        "by_op": [{"op": "decode_gemv", "time_ms": 5.0}],
                        "by_kernel_op": [{"kernel": "decode_fast"}],
                    }
                },
            },
        }]
    }), encoding="utf-8")
    rows = LAB.collect_operation_profiles([{
        "model_key": "m", "prompt_tokens": 32, "directory": str(directory),
    }])
    assert rows[0]["coverage_pct"] == 75.0
    assert rows[0]["top_operations"][0]["op"] == "gemm"
    assert rows[0]["selected_kernels"][0]["kernel"] == "gemm_fast"
    assert rows[0]["decode_coverage_pct"] == 75.0
    assert rows[0]["decode_core_equivalents"] == 3.0
    assert rows[0]["decode_selected_kernels"][0]["kernel"] == "decode_fast"


def test_perf_stat_counters_are_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "perf.csv"
    path.write_text(
        "1234,,cycles,100.00,100.00,\n42,,cache-misses,100.00,100.00,\n",
        encoding="utf-8",
    )
    assert LAB.parse_perf_stat_csv(path) == {"cycles": 1234.0, "cache-misses": 42.0}


def test_perf_stat_counters_aggregate_hybrid_core_types(tmp_path: Path) -> None:
    path = tmp_path / "perf.csv"
    path.write_text(
        "10,,cpu_core/cycles/,100.00,100.00,\n"
        "20,,cpu_atom/cycles/,100.00,100.00,\n",
        encoding="utf-8",
    )
    assert LAB.parse_perf_stat_csv(path) == {"cycles": 30.0}


def test_compare_rejects_runtime_below_requested_context(monkeypatch) -> None:
    model = COMPARE.ModelSpec("small", "Small", "Q8", Path("/m.gguf"), "small")
    monkeypatch.setattr(COMPARE, "ck_runtime_context_window", lambda _: 512)
    try:
        COMPARE.validate_runtime_capacity([model], 2048)
    except ValueError as exc:
        assert "capacity is 512" in str(exc)
        assert "requires 2048" in str(exc)
    else:
        raise AssertionError("undersized runtime was accepted")


def test_generated_context_capacity_reads_compiled_limit(tmp_path: Path) -> None:
    (tmp_path / "model_v8.c").write_text(
        "#define MAX_SEQ_LEN 2120\n", encoding="utf-8"
    )
    assert LAB.generated_context_capacity(tmp_path) == 2120
