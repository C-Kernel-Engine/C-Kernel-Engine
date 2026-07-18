#!/usr/bin/env python3
"""Run private numerical replay packets through fixed production oracles.

The manifest is data, never executable configuration. It may select only a
registered adapter; it cannot provide commands, environment variables, or
tolerances. Captured tensors remain outside the repository and are accepted
only after their complete files match the declared size and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "version/v8/schemas/numerical_replay_manifest.schema.json"
DEFAULT_REPORT = ROOT / "build/v8/numerical-replay/private-replay-report.json"
ADAPTER_ENVIRONMENT = {
    "CK_Q4K_Q8K_REAL_ACTIVATIONS_F32",
    "CK_Q4K_Q8K_REAL_WEIGHTS",
    "CK_Q4K_Q8K_REAL_EXPECTED_F32",
    "CK_Q4K_Q8K_REAL_WEIGHT_OFFSET",
    "CK_Q4K_Q8K_REAL_BIAS_OFFSET",
    "CK_Q4K_Q8K_REAL_M",
    "CK_Q4K_Q8K_REAL_N",
    "CK_Q4K_Q8K_REAL_K",
    "CK_Q6_XRAY_INPUT_F32",
    "CK_Q6_XRAY_WEIGHTS_Q6_K",
    "CK_Q6_XRAY_LLAMA_OUTPUT_F32",
    "CK_Q6_XRAY_N",
    "CK_Q6_XRAY_K",
}


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"{label} must be a JSON object")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path, "numerical replay manifest")
    schema = _load_json(SCHEMA, "numerical replay schema")
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ReplayError(f"manifest schema violation at {location}: {error.message}")

    fixture_ids = [fixture["id"] for fixture in manifest["fixtures"]]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ReplayError("fixture IDs must be unique")
    artifact_ids = set(manifest["artifacts"])
    for fixture in manifest["fixtures"]:
        missing = sorted(set(fixture["artifacts"].values()) - artifact_ids)
        if missing:
            raise ReplayError(f"fixture {fixture['id']} references unknown artifacts: {', '.join(missing)}")
        if fixture["adapter"] == "q4_k_q8_k_production" and "weight_offset" not in fixture:
            raise ReplayError(f"fixture {fixture['id']} requires weight_offset")
        if fixture["adapter"] == "q6_k_q8_k_production" and fixture["shape"]["m"] != 1:
            raise ReplayError(f"fixture {fixture['id']} requires m=1 for the Q6 production adapter")
        if fixture["adapter"] == "q6_k_q8_k_production" and (
            "weight_offset" in fixture or "bias_offset" in fixture
        ):
            raise ReplayError(f"fixture {fixture['id']} cannot use offsets with the Q6 production adapter")
    return manifest


def verify_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, VerifiedArtifact]:
    verified: dict[str, VerifiedArtifact] = {}
    for artifact_id, spec in manifest["artifacts"].items():
        raw = Path(spec["path"]).expanduser()
        path = raw if raw.is_absolute() else manifest_path.parent / raw
        path = path.resolve()
        if not path.is_file():
            raise ReplayError(f"artifact {artifact_id} is unavailable")
        actual_size = path.stat().st_size
        if actual_size != spec["size_bytes"]:
            raise ReplayError(
                f"artifact {artifact_id} size mismatch: expected {spec['size_bytes']}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != spec["sha256"]:
            raise ReplayError(f"artifact {artifact_id} SHA-256 mismatch")
        verified[artifact_id] = VerifiedArtifact(artifact_id, path, actual_hash, actual_size)
    return verified


def adapter_environment(
    fixture: dict[str, Any], artifacts: dict[str, VerifiedArtifact]
) -> dict[str, str]:
    refs = fixture["artifacts"]
    shape = fixture["shape"]
    activation = str(artifacts[refs["activations"]].path)
    weights = str(artifacts[refs["weights"]].path)
    expected = str(artifacts[refs["expected_output"]].path)
    if fixture["adapter"] == "q4_k_q8_k_production":
        env = {
            "CK_Q4K_Q8K_REAL_ACTIVATIONS_F32": activation,
            "CK_Q4K_Q8K_REAL_WEIGHTS": weights,
            "CK_Q4K_Q8K_REAL_EXPECTED_F32": expected,
            "CK_Q4K_Q8K_REAL_WEIGHT_OFFSET": str(fixture["weight_offset"]),
            "CK_Q4K_Q8K_REAL_M": str(shape["m"]),
            "CK_Q4K_Q8K_REAL_N": str(shape["n"]),
            "CK_Q4K_Q8K_REAL_K": str(shape["k"]),
        }
        if "bias_offset" in fixture:
            env["CK_Q4K_Q8K_REAL_BIAS_OFFSET"] = str(fixture["bias_offset"])
        return env
    if fixture["adapter"] == "q6_k_q8_k_production":
        return {
            "CK_Q6_XRAY_INPUT_F32": activation,
            "CK_Q6_XRAY_WEIGHTS_Q6_K": weights,
            "CK_Q6_XRAY_LLAMA_OUTPUT_F32": expected,
            "CK_Q6_XRAY_N": str(shape["n"]),
            "CK_Q6_XRAY_K": str(shape["k"]),
        }
    raise ReplayError(f"unsupported adapter: {fixture['adapter']}")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_suite(args: argparse.Namespace) -> int:
    report_path = args.report.resolve()
    if args.manifest is None:
        payload = {
            "schema_version": 1,
            "status": "SKIP",
            "reason": "CK_NUMERICAL_REPLAY_MANIFEST is unset",
            "fixtures": [],
        }
        _write_report(report_path, payload)
        print("Private numerical replay fixtures: SKIP (CK_NUMERICAL_REPLAY_MANIFEST is unset)")
        return 0

    manifest_path = args.manifest.resolve()
    try:
        manifest = load_manifest(manifest_path)
        artifacts = verify_artifacts(manifest_path, manifest)
        binaries = {
            "q4_k_q8_k_production": args.q4_binary.resolve(),
            "q6_k_q8_k_production": args.q6_binary.resolve(),
        }
        required_adapters = {fixture["adapter"] for fixture in manifest["fixtures"]}
        for adapter in required_adapters:
            if not binaries[adapter].is_file() or not os.access(binaries[adapter], os.X_OK):
                raise ReplayError(f"binary for adapter {adapter} is unavailable")

        results: list[dict[str, Any]] = []
        failed = False
        for fixture in manifest["fixtures"]:
            fixture_failed = False
            runs: list[dict[str, Any]] = []
            for threads in fixture["threads"]:
                env = os.environ.copy()
                for name in ADAPTER_ENVIRONMENT:
                    env.pop(name, None)
                env.update(adapter_environment(fixture, artifacts))
                env["CK_NUM_THREADS"] = str(threads)
                env["OMP_NUM_THREADS"] = "1"
                started = time.monotonic()
                try:
                    completed = subprocess.run(
                        [str(binaries[fixture["adapter"]]), "--quick"],
                        env=env,
                        check=False,
                    )
                except OSError as exc:
                    raise ReplayError(
                        f"fixture {fixture['id']} could not start adapter {fixture['adapter']}: {exc}"
                    ) from exc
                duration = time.monotonic() - started
                status = "PASS" if completed.returncode == 0 else "FAIL"
                fixture_failed |= completed.returncode != 0
                runs.append(
                    {
                        "threads": threads,
                        "status": status,
                        "returncode": completed.returncode,
                        "duration_seconds": round(duration, 6),
                    }
                )
            failed |= fixture_failed
            results.append(
                {
                    "id": fixture["id"],
                    "adapter": fixture["adapter"],
                    "status": "FAIL" if fixture_failed else "PASS",
                    "shape": fixture["shape"],
                    "artifact_sha256": {
                        role: artifacts[artifact_id].sha256
                        for role, artifact_id in fixture["artifacts"].items()
                    },
                    "runs": runs,
                }
            )
        payload = {
            "schema_version": 1,
            "suite_id": manifest["suite_id"],
            "status": "FAIL" if failed else "PASS",
            "fixtures": results,
        }
        _write_report(report_path, payload)
        print(f"Private numerical replay fixtures: {payload['status']} ({len(results)} fixtures)")
        return 1 if failed else 0
    except ReplayError as exc:
        _write_report(
            report_path,
            {"schema_version": 1, "status": "FAIL", "error": str(exc), "fixtures": []},
        )
        print(f"Private numerical replay fixtures: FAIL ({exc})", file=sys.stderr)
        return 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    manifest_default = os.environ.get("CK_NUMERICAL_REPLAY_MANIFEST")
    parser.add_argument("--manifest", type=Path, default=Path(manifest_default) if manifest_default else None)
    parser.add_argument("--q4-binary", type=Path, default=ROOT / "build/test_q4k_q8k_llama_packed")
    parser.add_argument("--q6-binary", type=Path, default=ROOT / "build/test_q6k_q8k_llama_production")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_suite(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
