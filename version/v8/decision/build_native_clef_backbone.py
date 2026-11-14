"""Author the pinned Clef backbone's explicit oneDNN 3.12 diagnostic profile.

Reuse the ordinary Qwen3.5 topology. Only projection numerical contracts change;
no generic lowering, resolver, provider or precision semantics are bypassed.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from version.v8.decision.capture_clef_features import (
    verify_lock,
    verify_release,
    file_sha,
)

OLD = "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output"
NEW = "bf16_weight_bf16_input_pytorch_onednn_3_12_brgemm_bf16_output"


def author_profile(template):
    result = copy.deepcopy(template)
    if result.get("name") != "qwen35":
        raise ValueError("pinned backbone topology required")
    changed = []
    for name, entry in result["required_numerical_contracts"].items():
        for phase, contract in entry.get("phases", {}).items():
            if contract.get("contract_id") == OLD:
                contract.update(
                    contract_id=NEW,
                    validation="observed",
                    evidence="Explicit oneDNN 3.12 projection provider; full Clef parity requires independent comparison.",
                )
                changed.append(phase)
    if sorted(changed) != ["decode", "prefill"]:
        raise ValueError("unexpected projection contract inventory")
    return result


def build(release, lock, lock_sha, output):
    import ck_run_v8 as driver

    release, output = Path(release), Path(output)
    if output.exists():
        raise ValueError(
            "use a new output directory; never reuse unbound converted weights"
        )
    verify_lock(release, lock, lock_sha)
    verify_release(release)
    output.mkdir(parents=True, exist_ok=False)
    _, _, manifest = driver.step_convert_safetensors(release, output)
    doc = json.loads(manifest.read_text())
    template = json.loads((ROOT / "version/v8/circuits/qwen35.json").read_text())
    doc["template"] = author_profile(template)
    manifest.write_text(json.dumps(doc, indent=2) + "\n")
    ir = driver.step_build_ir(
        manifest, output, context_len=512, logits_layout="last", refresh_circuit=False
    )
    source = driver.step_codegen(output, ir)
    driver.step_compile(source, output)
    provenance = dict(
        schema="cke.clef.native-backbone-build.v1",
        release_lock_sha256=lock_sha,
        manifest_sha256=file_sha(output / "weights_manifest.json"),
        projection_contract=NEW,
        numerical_parity="NOT_TESTED",
    )
    (output / "clef-build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    from version.v8.decision.run_native_clef import bind_bundle

    descriptor = output / "bundle.json"
    descriptor.write_text(json.dumps(bind_bundle(output, lock_sha), indent=2) + "\n")
    return file_sha(descriptor)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "lock", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--lock-sha256", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            dict(bundle_sha256=build(a.release, a.lock, a.lock_sha256, a.output))
        )
    )
