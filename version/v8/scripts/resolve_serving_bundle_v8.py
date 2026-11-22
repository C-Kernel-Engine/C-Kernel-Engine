#!/usr/bin/env python3
"""Resolve circuit-owned serving profiles into an offline deployment bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

V8_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V8_ROOT.parents[1]))
from server.serving_bundle import PROTOCOLS, SCHEMA, contract_identity, load_resolved_serving, sha256_file


def _json(path: Path) -> dict:
    doc = json.loads(path.read_bytes())
    if not isinstance(doc, dict):
        raise ValueError(f"expected an object: {path}")
    return doc


def _source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("profile reference must be relative")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping serving profile/asset: {relative}")
    return path


def _publisher_templates(run_dir: Path) -> dict[str, tuple[bytes, str]]:
    found: dict[str, tuple[bytes, str]] = {}

    def add(name: str, value: bytes, source: str) -> None:
        if not value.decode("utf-8").strip():
            raise ValueError(f"empty publisher template: {source}")
        if name in found and found[name][0] != value:
            raise ValueError(f"conflicting publisher template {name}: {source}")
        found[name] = (value, source)

    native = run_dir / "chat_template.jinja"
    if native.is_file():
        add("default", native.read_bytes(), "chat_template.jinja")
    # Legacy additional_chat_templates can contain operator-installed overrides.
    # Do not relabel them as publisher variants without an imported declaration.
    tokenizer = run_dir / "tokenizer_config.json"
    if tokenizer.is_file():
        templates = _json(tokenizer).get("chat_template")
        if isinstance(templates, str):
            add("default", templates.encode("utf-8"), "tokenizer_config.json:chat_template")
        elif isinstance(templates, dict):
            for name, value in templates.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise ValueError("invalid named publisher template")
                add(name, value.encode("utf-8"), f"tokenizer_config.json:{name}")
        elif isinstance(templates, list):
            for item in templates:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("template"), str):
                    raise ValueError("invalid publisher template variant")
                add(item["name"], item["template"].encode("utf-8"), f"tokenizer_config.json:{item['name']}")
        elif templates is not None:
            raise ValueError("invalid publisher chat_template")
    if "default" not in found:
        raise ValueError("publisher default template is missing")
    return found


def _verify_compilation_lineage(run_dir: Path, circuit: dict, *, allow_serving_update: bool) -> dict:
    """Verify retained manifest -> IR -> generated C -> runtime content identities.

    Stamps are build provenance, not signed attestations. Original absolute paths
    are retained for diagnostics; relocated bundles validate their local bytes.
    """
    manifest = _json(run_dir / "weights_manifest.json")
    if manifest.get("model") is not None and manifest["model"] != circuit.get("name"):
        raise ValueError("compiled manifest model does not match serving circuit")
    compiled = manifest.get("template")
    if not isinstance(compiled, dict):
        raise ValueError("missing compiled circuit snapshot; rebuild before metadata upgrade")
    if allow_serving_update:
        if {k: v for k, v in compiled.items() if k != "serving"} != {k: v for k, v in circuit.items() if k != "serving"}:
            raise ValueError("computational circuit differs from compiled circuit")
    elif compiled != circuit:
        raise ValueError("circuit differs from compiled circuit; serving-only changes require --allow-serving-update")

    def check(path: Path, identity: dict) -> None:
        if not isinstance(identity, dict) or identity.get("sha256") != sha256_file(path) or identity.get("size") != path.stat().st_size:
            raise ValueError(f"stale compilation provenance: {path.name}")

    stamps = {phase: _json(run_dir / f".ck_{phase}_bundle.json") for phase in ("ir", "codegen", "runtime")}
    schemas = {"ir": "ck-v8-ir-bundle-v1", "codegen": "ck-v8-codegen-bundle-v1", "runtime": "ck-v8-runtime-bundle-v2"}
    for phase, stamp in stamps.items():
        if stamp.get("inputs", {}).get("schema") != schemas[phase] or not stamp.get("outputs"):
            raise ValueError(f"invalid {phase} compilation provenance")
        for identity in stamp["outputs"].values():
            if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
                raise ValueError(f"invalid {phase} output identity")
            check(run_dir / Path(identity["path"]).name, identity)
    check(run_dir / "weights_manifest.json", stamps["ir"]["inputs"].get("manifest"))
    artifacts = stamps["codegen"]["inputs"].get("artifacts", {})
    for name, identity in stamps["ir"]["outputs"].items():
        if artifacts.get(name) != identity:
            raise ValueError(f"IR/codegen provenance mismatch: {name}")
    for identity in artifacts.values():
        check(run_dir / Path(identity["path"]).name, identity)
    model = stamps["codegen"]["outputs"].get("model_v8.c")
    if not model or stamps["runtime"]["inputs"].get("model_source") != model:
        raise ValueError("codegen/runtime provenance mismatch")
    for name in ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so"):
        check(run_dir / name, stamps["runtime"]["outputs"].get(name))
    return {"serving_only_update": compiled != circuit,
            "compiled_circuit_sha256": hashlib.sha256(json.dumps(compiled, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def resolve_runtime_dependencies(run_dir: Path, v8_root: Path = V8_ROOT) -> list[str]:
    """Resolve selected calls against the registry used for generated code."""
    registry_path = v8_root / "kernel_maps/KERNEL_REGISTRY.json"
    registry_hash = sha256_file(registry_path)
    sources = _json(run_dir / ".ck_codegen_bundle.json").get("inputs", {}).get("generator_sources", {})
    if not isinstance(sources, dict):
        raise ValueError("missing compiled kernel registry identity")
    compiled_hashes = [digest for path, digest in sources.items()
                       if Path(path).name == "KERNEL_REGISTRY.json"]
    if compiled_hashes != [registry_hash]:
        raise ValueError("kernel registry differs from compiled generated runtime; rebuild before inference")
    registry = _json(registry_path)
    kernels = registry.get("kernels")
    if not isinstance(kernels, list) or any(not isinstance(item, dict) for item in kernels):
        raise ValueError("invalid compiled kernel registry")
    maps = {item.get("_source_file"): item for item in kernels}
    if len(maps) != len(kernels):
        raise ValueError("ambiguous compiled kernel registry")
    dependencies: set[str] = set()
    for phase in ("prefill", "decode"):
        lowered = _json(run_dir / f"lowered_{phase}_call.json")
        operations = lowered.get("operations")
        if not isinstance(operations, list):
            raise ValueError(f"invalid {phase} lowered kernel calls")
        for operation in operations:
            call = operation.get("call_abi") if isinstance(operation, dict) else None
            if not isinstance(call, dict) or call.get("owner") != "kernel_map":
                continue
            source = call.get("source_file")
            if not isinstance(source, str) or Path(source).name != source or not source.endswith(".json"):
                raise ValueError("invalid selected kernel map source")
            mapping = maps.get(source)
            if not isinstance(mapping, dict):
                raise ValueError(f"missing compiled kernel map: {source}")
            implementation = mapping.get("impl")
            if (mapping.get("id") != call.get("kernel_id")
                    or not isinstance(implementation, dict)
                    or implementation.get("function") != operation.get("function")):
                raise ValueError(f"selected kernel map changed after compilation: {source}")
            declared = mapping.get("runtime_dependencies", [])
            if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
                raise ValueError(f"invalid runtime dependencies: {source}")
            dependencies.update(declared)
    return sorted(dependencies)


def resolve_serving_bundle(run_dir: Path, circuit_path: Path, *, variant: str | None = None, v8_root: Path = V8_ROOT, allow_serving_update: bool = False) -> dict:
    run_dir, circuit_path, v8_root = Path(run_dir), Path(circuit_path), Path(v8_root)
    circuit = _json(circuit_path)
    declaration = circuit.get("serving")
    if not isinstance(declaration, dict) or declaration.get("schema") != "cke.circuit_serving.v1":
        raise ValueError("circuit has no supported serving declaration")
    profile_ref = declaration.get("profile_ref")
    profile_path = _source_path(v8_root, profile_ref)
    profile = _json(profile_path)
    if profile.get("schema") != "cke.serving_profile.v1" or profile.get("renderer") != "jinja-chat-v1" or profile.get("input_modalities") != ["text"]:
        raise ValueError("unsupported serving profile contract")
    selected = variant or declaration.get("default_variant")
    spec = profile.get("variants", {}).get(selected)
    if not isinstance(spec, dict) or spec.get("output_protocol") not in PROTOCOLS:
        raise ValueError(f"unsupported serving variant: {selected}")
    stop_text = spec.get("stop_text", [])
    if (not isinstance(stop_text, list) or any(not isinstance(item, str) or not item for item in stop_text)
            or len(set(stop_text)) != len(stop_text)):
        raise ValueError(f"invalid serving stop_text for variant: {selected}")
    message_role_map = spec.get("message_role_map", {})
    if not isinstance(message_role_map, dict) or any(
        source != "developer" or target != "system"
        for source, target in message_role_map.items()
    ):
        raise ValueError(f"invalid serving message_role_map for variant: {selected}")
    config = _json(run_dir / "config.json")
    # GGUF conversion uses model_type/model_name; only some runtime configs
    # declare a canonical model key. The retained compiled manifest and its
    # verified circuit snapshot are the authoritative identity in both cases.
    if config.get("model") is not None and config["model"] != circuit.get("name"):
        raise ValueError("circuit does not match runtime configuration")
    if config.get("model") is None and _json(run_dir / "weights_manifest.json").get("model") != circuit.get("name"):
        raise ValueError("circuit does not match compiled manifest model")
    lineage = _verify_compilation_lineage(run_dir, circuit, allow_serving_update=allow_serving_update)
    runtime_dependencies = resolve_runtime_dependencies(run_dir, v8_root)
    publisher = _publisher_templates(run_dir)
    assets: dict[str, dict] = {}

    def store(name: str, data: bytes, provenance: str, suffix: str) -> None:
        digest = hashlib.sha256(data).hexdigest()
        relative = f"serving/assets/{digest}{suffix}"
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError("conflicting content-addressed serving asset")
        if not path.exists():
            path.write_bytes(data)
        assets[name] = {"path": relative, "sha256": digest, "source": provenance}

    store("circuit", circuit_path.read_bytes(), "circuit-template", ".json")
    store("profile", profile_path.read_bytes(), profile_ref, ".json")
    for role in ("chat", "tools"):
        binding = spec.get(role)
        if not isinstance(binding, dict):
            raise ValueError(f"missing template binding: {role}")
        if binding.get("source") == "publisher":
            name = binding.get("variant")
            if name not in publisher and "fallback_variant" in binding:
                name = binding["fallback_variant"]
            if name not in publisher:
                raise ValueError(f"missing publisher template variant: {name}")
            data, source = publisher[name]
        elif binding.get("source") == "profile_asset":
            path = _source_path(profile_path.parent, binding.get("path"))
            data, source = path.read_bytes(), f"{profile_ref}:{binding['path']}"
        else:
            raise ValueError(f"unsupported template source: {role}")
        if not data.decode("utf-8").strip():
            raise ValueError(f"empty selected template: {role}")
        store(role, data, source, ".jinja")
    required = ("config.json", "layout_decode.json", "libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so", "weights.bump", "weights_manifest.map")
    optional = ("weights_manifest.json", ".ck_ir_bundle.json", ".ck_codegen_bundle.json", ".ck_runtime_bundle.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "tiktoken.model", "generation_config.json")
    for name in required + optional:
        path = run_dir / name
        if name in required or path.is_file():
            assets[name] = {"path": name, "sha256": sha256_file(path), "source": "generated-or-imported-bundle"}
    capacity = _json(run_dir / "layout_decode.json").get("config", {}).get("context_length")
    document = {"schema": SCHEMA, "profile_id": profile.get("id"), "profile_ref": profile_ref, "variant": selected,
                "renderer": profile["renderer"], "output_protocol": spec["output_protocol"],
                "stop_text": stop_text,
                "message_role_map": message_role_map,
                "input_modalities": profile["input_modalities"], "context_capacity": capacity,
                "assets": assets, "compilation_lineage": lineage,
                "runtime_dependencies": runtime_dependencies, "certification": "NOT_TESTED"}
    document["identity"] = contract_identity(document)
    load_resolved_serving(run_dir, document=document)
    # Publish last: an interrupted asset write cannot publish a new contract.
    sidecar = run_dir / "serving.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                     prefix=".serving-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(document, indent=2) + "\n")
    try:
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--allow-serving-update", action="store_true",
                        help="Permit only the serving section to differ from the compiled circuit snapshot")
    args = parser.parse_args()
    doc = resolve_serving_bundle(args.run, args.circuit, variant=args.variant, allow_serving_update=args.allow_serving_update)
    print(f"Resolved serving bundle {doc['identity']} ({doc['variant']}); certification NOT_TESTED")


if __name__ == "__main__":
    main()
