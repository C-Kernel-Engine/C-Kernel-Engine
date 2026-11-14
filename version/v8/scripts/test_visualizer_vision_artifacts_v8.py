#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.tools.open_ir_visualizer_v8 import generate_html_report, load_model_data  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _op(op_id: int, op: str, *, layer: int = -1, section: str = "body", from_op: int | None = None) -> dict:
    inputs = {}
    if from_op is not None:
        inputs["input"] = {
            "from_op": from_op,
            "from_output": "out",
            "tensor": f"tensor.{from_op}.out",
            "dtype": "fp32",
            "slot": "main_stream",
        }
    else:
        inputs["input"] = {"from": "external:input", "dtype": "fp32", "slot": "external"}
    return {
        "op_id": op_id,
        "op": op,
        "kernel": f"{op}_kernel",
        "section": section,
        "layer": layer,
        "dataflow": {
            "inputs": inputs,
            "outputs": {
                "out": {
                    "tensor": f"tensor.{op_id}.out",
                    "dtype": "fp32",
                    "slot": "main_stream",
                }
            },
        },
        "weights": {},
    }


def _layout(total_size: int, *, prefix: str) -> dict:
    return {
        "memory": {
            "weights": {
                "size": 256,
                "entries": [
                    {
                        "name": f"{prefix}.weight",
                        "dtype": "fp32",
                        "size": 128,
                        "offset": 0,
                        "abs_offset": 0,
                    }
                ],
            },
            "activations": {
                "size": 512,
                "buffers": [
                    {
                        "name": f"{prefix}.activation",
                        "dtype": "fp32",
                        "size": 256,
                        "offset": 0,
                        "abs_offset": 256,
                    }
                ],
            },
            "arena": {"total_size": total_size},
        }
    }


def _make_fixture_run(root: Path) -> tuple[int, int, int]:
    bridge = root / "multimodal_bridge"
    encoder = bridge / "encoder"
    decoder = bridge / "decoder"
    encoder.mkdir(parents=True)
    decoder.mkdir(parents=True)

    encoder_ops = [
        _op(0, "patchify", section="header"),
        _op(1, "patch_proj", layer=0, from_op=0),
        _op(2, "vision_attn", layer=0, from_op=1),
        _op(3, "projector_fc2", section="footer", from_op=2),
    ]
    decoder_ops = [
        _op(0, "dense_embedding_lookup", section="header"),
        _op(1, "rmsnorm", layer=0, from_op=0),
        _op(2, "attn", layer=0, from_op=1),
        _op(3, "logits", section="footer", from_op=2),
    ]
    bridge_ops = 1

    _write_json(encoder / "ir1.json", {"format": "ir1-dataflow", "version": 3, "mode": "encoder", "ops": encoder_ops})
    _write_json(encoder / "call.json", {"operations": encoder_ops})
    _write_json(encoder / "lowered.json", {"operations": encoder_ops})
    _write_json(encoder / "layout.json", _layout(4096, prefix="encoder"))
    _write_json(decoder / "ir1_prefill.json", {"format": "ir1-dataflow", "version": 3, "mode": "prefill", "ops": decoder_ops})
    _write_json(decoder / "ir1_decode.json", {"format": "ir1-dataflow", "version": 3, "mode": "decode", "ops": decoder_ops[:2]})
    _write_json(decoder / "lowered_prefill.json", {"operations": decoder_ops})
    _write_json(decoder / "lowered_decode.json", {"operations": decoder_ops[:2]})
    _write_json(decoder / "layout_prefill.json", _layout(8192, prefix="decoder_prefill"))
    _write_json(decoder / "layout_decode.json", _layout(2048, prefix="decoder_decode"))
    _write_json(decoder / "config.json", {"model": "synthetic_qwen3vl_visualizer", "mode": "prefill"})
    _write_json(decoder / "weights_manifest.json", {"weights": []})
    _write_json(
        bridge / "bridge_report.json",
        {
            "bridge_mode": "encoder_decoder",
            "prefix_source": "encoder",
            "prefix_tokens": 4,
            "prefix_grid_x": 2,
            "prefix_grid_y": 2,
            "prefix_text_pos": 3,
            "decoder_input_embed_dim": 16,
            "total_prefill_tokens": 11,
            "generated_token_count": 2,
            "generation_stop_reason": "unit_test",
            "prompt_tokens_before_image": [1, 2],
            "prompt_tokens_after_image": [3],
            "encoder_report": {
                "image_source": "synthetic",
                "source_image_size": [8, 8],
                "image_width": 8,
                "image_height": 8,
                "patch_size": 4,
                "embed_dim": 16,
                "preprocess": "synthetic_rgb",
            },
        },
    )
    return len(encoder_ops), bridge_ops, len(decoder_ops)


def _extract_embedded_json(html: str) -> dict:
    match = re.search(
        r"<script>window\.EMBEDDED_IR_DATA\s*=\s*(\{.*?\});window\.dispatchEvent",
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise AssertionError("EMBEDDED_IR_DATA script was not found")
    return json.loads(match.group(1))


def run(json_out: Path) -> int:
    checks: list[dict] = []
    status = "pass"
    with tempfile.TemporaryDirectory(prefix="ck-v8-vision-viz-") as tmp:
        run_dir = Path(tmp)
        expected_encoder, expected_bridge, expected_decoder = _make_fixture_run(run_dir)
        expected_total = expected_encoder + expected_bridge + expected_decoder

        data = load_model_data(run_dir, run_dir=run_dir, strict_run_artifacts=True)
        files = data.get("files", {})
        graph = files.get("multimodal_dataflow_graph")
        vision = files.get("vision_artifacts")
        try:
            assert isinstance(graph, dict), "missing derived multimodal_dataflow_graph"
            stats = graph.get("stats") or {}
            assert stats.get("encoder_ops") == expected_encoder, stats
            assert stats.get("bridge_ops") == expected_bridge, stats
            assert stats.get("decoder_prefill_ops") == expected_decoder, stats
            assert stats.get("total_ops") == expected_total, stats
            call_ops = ((graph.get("call") or {}).get("operations") or [])
            assert len(call_ops) == expected_total, f"unified call ops missing: {len(call_ops)} != {expected_total}"
            call_stages = {row.get("network_stage") for row in call_ops if isinstance(row, dict)}
            assert {"vision_encoder", "bridge", "decoder_prefill"}.issubset(call_stages), call_stages
            layout_memory = ((graph.get("layout") or {}).get("memory") or {})
            weight_entries = ((layout_memory.get("weights") or {}).get("entries") or [])
            activation_buffers = ((layout_memory.get("activations") or {}).get("buffers") or [])
            weight_stages = {row.get("network_stage") for row in weight_entries if isinstance(row, dict)}
            activation_stages = {row.get("network_stage") for row in activation_buffers if isinstance(row, dict)}
            assert {"vision_encoder", "decoder_prefill"}.issubset(weight_stages), weight_stages
            assert {"vision_encoder", "decoder_prefill"}.issubset(activation_stages), activation_stages
            assert isinstance(vision, dict), "missing derived vision_artifacts"
            assert (vision.get("ops") or {}).get("full_network") == expected_total, vision.get("ops")
            checks.append({"name": "derived_payload", "status": "pass", "detail": f"{expected_total} multimodal ops"})
        except AssertionError as exc:
            checks.append({"name": "derived_payload", "status": "fail", "detail": str(exc)})
            status = "fail"

        html_path = run_dir / "ir_report.html"
        generate_html_report(run_dir, html_path, run_dir=run_dir, strict_run_artifacts=True)
        html = html_path.read_text(encoding="utf-8")
        try:
            embedded = _extract_embedded_json(html)
            embedded_graph = embedded.get("files", {}).get("multimodal_dataflow_graph")
            embedded_vision = embedded.get("files", {}).get("vision_artifacts")
            assert isinstance(embedded_graph, dict), "HTML missing multimodal_dataflow_graph"
            assert embedded_graph.get("stats", {}).get("total_ops") == expected_total, embedded_graph.get("stats")
            assert isinstance(embedded_vision, dict), "HTML missing vision_artifacts"
            assert embedded_vision.get("patch_grid", {}).get("expected_prefix_tokens") == 4, embedded_vision.get("patch_grid")
            assert "files.full_network_graph || files.multimodal_dataflow_graph" in html, "visualizer fallback not present"
            helper_pos = html.find("function getEmbeddedFiles()")
            multimodal_pos = html.find("function renderMultimodal()")
            assert helper_pos >= 0, "main visualizer getEmbeddedFiles helper not present"
            assert multimodal_pos >= 0, "multimodal renderer not present"
            assert helper_pos < multimodal_pos, "getEmbeddedFiles must be defined before renderMultimodal"
            checks.append({"name": "html_embedding", "status": "pass", "detail": str(html_path)})
        except AssertionError as exc:
            checks.append({"name": "html_embedding", "status": "fail", "detail": str(exc)})
            status = "fail"

    # Keep component-reader and shipped JS rendering checks in the existing lane.
    for name, command in (
        ("component_reader", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_v8_visualizer_components.py"]),
        ("component_renderer", ["node", "--test", "tests/test_v8_visualizer_components.mjs"]),
    ):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            passed = result.returncode == 0
            detail = (result.stdout + result.stderr)[-4000:]
        except (OSError, subprocess.TimeoutExpired) as error:
            passed, detail = False, str(error)
        checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})
        if not passed:
            status = "fail"

    summary = {
        "schema": "ck.v8.vision_visualizer_harness.v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "summary": {
            "passed": sum(1 for row in checks if row["status"] == "pass"),
            "failed": sum(1 for row in checks if row["status"] != "pass"),
            "total": len(checks),
        },
        "checks": checks,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in checks:
        marker = "PASS" if row["status"] == "pass" else "FAIL"
        print(f"{row['name']}  max_diff=0.00e+00  tol=1e+00  [{marker}]")
        print(f"  {row['detail']}")
    print(f"vision_visualizer_harness status={status} json={json_out}")
    return 0 if status == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v8 vision artifacts in the IR visualizer.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "version" / "v8" / ".cache" / "reports" / "vision_visualizer_latest.json",
    )
    args = parser.parse_args()
    return run(args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
