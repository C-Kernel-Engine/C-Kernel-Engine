"""Validate resolved serving data without consulting repository model profiles."""
from __future__ import annotations

import hashlib
import ctypes
import json
import os
import re
from pathlib import Path
from typing import Any

PROTOCOLS = frozenset({"none", "tagged_json", "bare_json", "qwen_xml", "qwen_code_xml", "qwen_code_xml_raw_v2", "gemma4_dsl_v1", "muse_atem_v1", "nemotron_toolcall_v1"})
SCHEMA = "cke.resolved_serving.v1"
RUNTIME_DEPENDENCIES = {
    "mkl_sgemm": ("CK_MKL_LIBRARY", "sgemm_"),
    "mkl_vsexp": ("CK_MKL_LIBRARY", "vsExp"),
    "sleef_expf16_u10": ("CK_SLEEF_LIBRARY", "Sleef_expf16_u10"),
    "sleef_log1pf16_u10": ("CK_SLEEF_LIBRARY", "Sleef_log1pf16_u10"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_loaded_symbol_backing(
    library: ctypes.CDLL, symbol: str, expected_path: Path,
) -> dict[str, str]:
    """Identify bytes backing an executable native symbol, not its loader filename."""
    try:
        address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p).value
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot inspect loaded symbol {symbol}") from exc
    if not address:
        raise ValueError(f"loaded symbol has no address: {symbol}")
    matching = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        lo, hi = (int(part, 16) for part in fields[0].split("-", 1))
        if lo <= address < hi:
            matching.append(fields)
    if len(matching) != 1:
        raise ValueError(f"loaded symbol has no unique executable mapping: {symbol}")
    fields = matching[0]
    mapped = fields[5]
    if mapped.endswith(" (deleted)") or Path(mapped).resolve() != expected_path.resolve():
        raise ValueError(f"loaded symbol backing path changed: {symbol}")
    try:
        with Path(mapped).open("rb") as stream:
            backing = os.fstat(stream.fileno())
            device = f"{os.major(backing.st_dev):02x}:{os.minor(backing.st_dev):02x}"
            if fields[3].lower() != device or int(fields[4]) != backing.st_ino:
                raise ValueError(f"loaded symbol backing file was replaced: {symbol}")
            digest = hashlib.sha256()
            for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(part)
            current = Path(mapped).stat()
            if current.st_dev != backing.st_dev or current.st_ino != backing.st_ino:
                raise ValueError(f"loaded symbol backing file was replaced: {symbol}")
    except OSError as exc:
        raise ValueError(f"cannot read loaded symbol backing file: {symbol}") from exc
    return {"method": "symbol_mapping_dev_inode_sha256", "symbol": symbol,
            "sha256": digest.hexdigest()}


def contract_identity(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "identity"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bundle_path(run_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("serving asset path must be bundle-relative")
    path = run_dir / relative
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError(f"serving asset escapes bundle: {relative}")
    return path


def load_resolved_serving(run_dir: Path, *, document: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return validated data; an invalid present sidecar never falls back."""
    run_dir = Path(run_dir)
    sidecar = run_dir / "serving.json"
    if document is None and not sidecar.exists() and not sidecar.is_symlink():
        return None  # Explicit legacy bundle support, not circuit certification.
    try:
        doc = document if document is not None else json.loads(sidecar.read_bytes())
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ValueError("invalid resolved serving schema")
        if doc.get("identity") != contract_identity(doc):
            raise ValueError("resolved serving identity mismatch")
        if doc.get("renderer") != "jinja-chat-v1":
            raise ValueError("unsupported serving renderer")
        if doc.get("output_protocol") not in PROTOCOLS:
            raise ValueError("unsupported serving output protocol")
        if doc.get("input_modalities") != ["text"]:
            raise ValueError("this resolved serving path supports text only")
        dependencies = doc.get("runtime_dependencies", [])
        if (not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies))
                or any(item not in RUNTIME_DEPENDENCIES for item in dependencies)):
            raise ValueError("unsupported serving runtime dependencies")
        assets = doc.get("assets")
        if not isinstance(assets, dict) or not assets:
            raise ValueError("missing serving asset inventory")
        required = {"circuit", "profile", "chat", "tools", "config.json", "layout_decode.json", "libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so", "weights.bump", "weights_manifest.map"}
        if not required.issubset(assets):
            raise ValueError("incomplete serving asset inventory")
        for name, item in assets.items():
            if not isinstance(item, dict):
                raise ValueError(f"invalid serving asset: {name}")
            path = bundle_path(run_dir, item.get("path"))
            if name.endswith((".json", ".so", ".bump", ".model", ".map")) and name not in {"circuit", "profile", "chat", "tools"} and item.get("path") != name:
                raise ValueError(f"serving runtime asset has wrong binding: {name}")
            if sha256_file(path) != item.get("sha256"):
                raise ValueError(f"stale serving asset: {name}")
        circuit = json.loads(bundle_path(run_dir, assets["circuit"]["path"]).read_bytes())
        profile = json.loads(bundle_path(run_dir, assets["profile"]["path"]).read_bytes())
        config = json.loads(bundle_path(run_dir, assets["config.json"]["path"]).read_bytes())
        layout = json.loads(bundle_path(run_dir, assets["layout_decode.json"]["path"]).read_bytes())
        declaration = circuit.get("serving", {})
        if declaration.get("schema") != "cke.circuit_serving.v1" or declaration.get("profile_ref") != doc.get("profile_ref"):
            raise ValueError("serving circuit/profile ownership mismatch")
        if config.get("model") is not None and config["model"] != circuit.get("name"):
            raise ValueError("serving circuit does not match runtime configuration")
        if config.get("model") is None:
            manifest_asset = assets.get("weights_manifest.json")
            if manifest_asset is None:
                raise ValueError("serving circuit has no compiled manifest identity")
            manifest = json.loads(bundle_path(run_dir, manifest_asset["path"]).read_bytes())
            if manifest.get("model") != circuit.get("name"):
                raise ValueError("serving circuit does not match compiled manifest model")
        if profile.get("schema") != "cke.serving_profile.v1" or profile.get("id") != doc.get("profile_id"):
            raise ValueError("serving profile identity mismatch")
        variant = profile.get("variants", {}).get(doc.get("variant"))
        if not isinstance(variant, dict) or variant.get("output_protocol") != doc["output_protocol"]:
            raise ValueError("serving variant/protocol mismatch")
        if doc.get("stop_text", []) != variant.get("stop_text", []):
            raise ValueError("serving variant/stop policy mismatch")
        message_role_map = doc.get("message_role_map", {})
        if (message_role_map != variant.get("message_role_map", {})
                or not isinstance(message_role_map, dict)
                or any(source != "developer" or target != "system"
                       for source, target in message_role_map.items())):
            raise ValueError("serving variant/message-role policy mismatch")
        stop_text = doc.get("stop_text", [])
        if (not isinstance(stop_text, list) or any(not isinstance(item, str) or not item for item in stop_text)
                or len(set(stop_text)) != len(stop_text)):
            raise ValueError("invalid serving stop policy")
        if profile.get("renderer") != doc["renderer"] or profile.get("input_modalities") != doc["input_modalities"]:
            raise ValueError("serving profile capabilities mismatch")
        capacity = layout.get("config", {}).get("context_length")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0 or capacity != doc.get("context_capacity"):
            raise ValueError("serving context capacity mismatch")
        for role in ("chat", "tools"):
            if not bundle_path(run_dir, assets[role]["path"]).read_bytes().decode("utf-8").strip():
                raise ValueError(f"empty serving template: {role}")
        return doc
    except (OSError, UnicodeError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resolved serving bundle {sidecar}: {exc}") from exc


def verify_runtime_dependencies(doc: dict[str, Any], *, native_library: Path | None = None) -> None:
    """Reject missing lazy native dependencies before starting inference."""
    missing = []
    libraries: dict[str | Path | None, ctypes.CDLL | None] = {}

    def exports(path: str | Path | None, symbol: str) -> bool:
        if path not in libraries:
            try:
                libraries[path] = ctypes.CDLL(path)
            except OSError:
                libraries[path] = None
        return getattr(libraries[path], symbol, None) is not None

    for dependency in doc.get("runtime_dependencies", []):
        if dependency not in RUNTIME_DEPENDENCIES:
            raise ValueError(f"unsupported native runtime dependency: {dependency}")
        variable, symbol = RUNTIME_DEPENDENCIES[dependency]
        path = os.environ.get(variable) or None
        if exports(path, symbol):
            continue
        if path is None and native_library is not None and exports(native_library, symbol):
            continue
        missing.append(f"{symbol} (set {variable} to a library that exports it)")
    if missing:
        raise ValueError("selected native kernels require " + ", ".join(missing))


def resolved_templates(run_dir: Path, doc: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    assets = doc["assets"]
    chat = bundle_path(run_dir, assets["chat"]["path"]).read_bytes().decode("utf-8")
    tools = bundle_path(run_dir, assets["tools"]["path"]).read_bytes().decode("utf-8")
    return chat, {"tool_use": tools}, {"message_role_map": doc.get("message_role_map", {})}


def resolved_renderer_tokens(
    run_dir: Path, doc: dict[str, Any], *, decode_ids=None, lookup_token=None,
) -> dict[str, str]:
    """Read publisher Jinja token variables only from verified bundle assets.

    The caller must first validate ``doc`` with ``load_resolved_serving``. A
    missing variable stays undefined so a template that requires it fails
    rather than rendering an invented delimiter.
    """
    variables: dict[str, str] = {}

    def bind(key: str, value: str) -> None:
        if not value:
            raise ValueError(f"invalid publisher token variable: {key}")
        if key in variables and variables[key] != value:
            raise ValueError(f"conflicting publisher token variable: {key}")
        variables[key] = value

    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        asset = doc["assets"].get(name)
        if asset is None:
            continue
        payload = json.loads(bundle_path(Path(run_dir), asset["path"]).read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid publisher token metadata: {name}")
        for key, value in payload.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_token", key):
                continue
            if isinstance(value, dict):
                value = value.get("content")
            if value is None or not isinstance(value, str):
                # An unsupported metadata shape cannot supply a Jinja scalar;
                # templates requiring it still fail under StrictUndefined.
                continue
            bind(key, value)

    # A GGUF conversion can retain special-token IDs in its verified manifest
    # without HF token metadata. Prefer the verified tokenizer.json table;
    # otherwise decode the verified IDs with the already-loaded model tokenizer.
    manifest_asset = doc["assets"].get("weights_manifest.json")
    tokenizer_asset = doc["assets"].get("tokenizer.json")
    if manifest_asset is not None:
        manifest = json.loads(bundle_path(Path(run_dir), manifest_asset["path"]).read_bytes())
        tokenizer = (json.loads(bundle_path(Path(run_dir), tokenizer_asset["path"]).read_bytes())
                     if tokenizer_asset is not None else {})
        if not isinstance(manifest, dict) or not isinstance(tokenizer, dict):
            raise ValueError("invalid bundled GGUF token metadata")
        special = manifest.get("special_tokens", {})
        added = tokenizer.get("added_tokens", [])
        if not isinstance(special, dict) or not isinstance(added, list):
            raise ValueError("invalid bundled GGUF special-token metadata")
        by_id: dict[int, str] = {}
        for item in added:
            if not isinstance(item, dict):
                continue
            token_id, content = item.get("id"), item.get("content")
            if (not isinstance(token_id, int) or isinstance(token_id, bool)
                    or not isinstance(content, str)):
                continue
            if token_id in by_id and by_id[token_id] != content:
                raise ValueError(f"conflicting bundled tokenizer text for ID {token_id}")
            by_id[token_id] = content
        for name in ("bos", "eos", "unk", "pad"):
            token_id = special.get(f"{name}_token_id")
            if isinstance(token_id, int) and not isinstance(token_id, bool):
                value = by_id.get(token_id)
                if value is not None and lookup_token is not None:
                    try:
                        loaded_id = lookup_token(value)
                    except Exception as exc:
                        raise ValueError(
                            f"loaded tokenizer cannot look up verified {name} token"
                        ) from exc
                    if loaded_id != token_id:
                        raise ValueError(f"loaded tokenizer disagrees with bundled {name} token")
                elif decode_ids is not None:
                    try:
                        loaded_value = decode_ids([token_id])
                    except Exception as exc:
                        raise ValueError(
                            f"loaded tokenizer cannot decode verified {name} token ID {token_id}"
                        ) from exc
                    if value is not None and value != loaded_value:
                        raise ValueError(f"loaded tokenizer disagrees with bundled {name} token")
                    value = loaded_value
                # Display detokenization can suppress a control token. Its ID
                # remains valid, but an empty rendering cannot supply a Jinja
                # variable; templates that need it fail under StrictUndefined.
                if value:
                    bind(f"{name}_token", value)
    return variables


def verify_loaded_libraries(doc: dict[str, Any], *, maps_text: str | None = None) -> None:
    """Bind file validation to actual Linux native dependencies after session open."""
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text()
        except OSError as exc:
            raise ValueError("resolved serving requires Linux loaded-library identity inspection") from exc
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.add(fields[5])
    for name in ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so"):
        loaded = {Path(path) for path in paths if Path(path.removesuffix(" (deleted)")).name == name}
        if not loaded:
            raise ValueError(f"required serving library is not loaded: {name}")
        expected = doc["assets"][name]["sha256"]
        for path in loaded:
            try:
                matches = sha256_file(path) == expected
            except OSError as exc:
                raise ValueError(f"cannot identify loaded serving library: {path}") from exc
            if not matches:
                raise ValueError(f"loaded serving library identity mismatch: {name}")


def loaded_serving_identity(
    doc: dict[str, Any], *, model: str, session_library: ctypes.CDLL,
) -> dict[str, Any]:
    """Describe a verified loaded bundle without publishing filesystem paths.

    Call only after ``verify_loaded_libraries`` succeeds on the open session.
    The session ABI library is outside the BUMP bundle, so identify its loaded
    file separately instead of implying that the bundle pins it.
    """
    try:
        symbol_address = ctypes.cast(session_library.ck_session_v8_open, ctypes.c_void_p).value
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("cannot identify loaded native session symbol") from exc
    if not symbol_address:
        raise ValueError("cannot identify loaded native session symbol")
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError as exc:
        raise ValueError("cannot inspect loaded native session mapping") from exc
    candidates = []
    selected = []
    for row in maps.splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if Path(path.removesuffix(" (deleted)")).name != "libck_session_v8.so":
            continue
        start, end = (int(part, 16) for part in fields[0].split("-", 1))
        entry = (fields, path)
        candidates.append(entry)
        if start <= symbol_address < end and "x" in fields[1]:
            selected.append(entry)
    if len(selected) != 1 or not candidates:
        raise ValueError("loaded native session symbol has no unique executable mapping")
    fields, path = selected[0]
    if path.endswith(" (deleted)") or any(
        item[0][3:5] != fields[3:5] for item in candidates if item[1] == path
    ):
        raise ValueError("loaded native session mapping is deleted or ambiguous")
    library = Path(path)
    try:
        with library.open("rb") as stream:
            backing = os.fstat(stream.fileno())
            device = f"{os.major(backing.st_dev):02x}:{os.minor(backing.st_dev):02x}"
            if fields[3].lower() != device or int(fields[4]) != backing.st_ino:
                raise ValueError("loaded native session backing file was replaced")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
            current = library.stat()
            if current.st_dev != backing.st_dev or current.st_ino != backing.st_ino:
                raise ValueError("loaded native session backing file was replaced")
    except OSError as exc:
        raise ValueError("cannot read loaded native session backing file") from exc
    return {
        "schema": "cke.loaded_serving_identity.v1",
        "model": model,
        "serving_identity": doc["identity"],
        "profile_id": doc["profile_id"],
        "variant": doc["variant"],
        "output_protocol": doc["output_protocol"],
        "context_capacity": doc["context_capacity"],
        "assets_sha256": {
            name: asset["sha256"] for name, asset in sorted(doc["assets"].items())
        },
        "session_library_sha256": digest.hexdigest(),
    }


def _manifest_special_tokens(run_dir: Path) -> dict[str, Any]:
    """Read the special_tokens document from runtime artifacts (manifest first)."""
    for name in ("weights_manifest.json", "init.json"):
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        special = doc.get("special_tokens")
        if isinstance(special, dict) and special:
            return special
    return {}


def _manifest_special_token_ids(run_dir: Path) -> dict[str, int]:
    """Read special-token IDs from runtime artifacts (manifest first)."""
    ids: dict[str, int] = {}
    for key, value in _manifest_special_tokens(run_dir).items():
        if (isinstance(key, str) and key.endswith("_token_id")
                and isinstance(value, int) and not isinstance(value, bool)):
            ids[key[: -len("_id")]] = value
    return ids


def _loaded_model_token_decoder(run_dir: Path):
    """Return an ID->text decoder bound to the already-loaded runtime library.

    The native session dlopens the runtime's model library; attaching with
    RTLD_NOLOAD reuses that mapping instead of initializing a second model.
    Returns None when the runtime library is not loaded in this process.
    """
    if not hasattr(os, "RTLD_NOLOAD"):
        return None
    lib = None
    for name in ("libmodel.so", "ck-kernel-inference.so", "ck-kernel-decode.so"):
        try:
            lib = ctypes.CDLL(
                str(Path(run_dir).resolve() / name),
                mode=os.RTLD_NOLOAD | os.RTLD_NOW,
            )
            break
        except OSError:
            continue
    if lib is None:
        return None
    try:
        decode = lib.ck_model_decode_tokens
    except AttributeError:
        return None
    decode.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    decode.restype = ctypes.c_int

    def decode_ids(ids: list[int]) -> str:
        arr = (ctypes.c_int32 * len(ids))(*ids)
        buf = ctypes.create_string_buffer(max(64, len(ids) * 16))
        out_len = decode(arr, len(ids), buf, len(buf))
        if out_len < 0:
            raise ValueError("runtime tokenizer decode failed")
        return bytes(buf.raw[:out_len]).decode("utf-8", errors="replace")

    return decode_ids


def _loaded_model_token_lookup(run_dir: Path):
    """Look up raw token spelling in the already-loaded model tokenizer.

    Display decoding can intentionally suppress BOS, EOS and PAD, so it cannot
    verify their publisher spellings. The model's exact vocabulary lookup can.
    """
    if not hasattr(os, "RTLD_NOLOAD"):
        return None
    for name in ("libmodel.so", "ck-kernel-inference.so", "ck-kernel-decode.so"):
        try:
            lib = ctypes.CDLL(
                str(Path(run_dir).resolve() / name),
                mode=os.RTLD_NOLOAD | os.RTLD_NOW,
            )
        except OSError:
            continue
        try:
            lookup = lib.ck_model_lookup_token
        except AttributeError:
            continue
        lookup.argtypes = [ctypes.c_char_p]
        lookup.restype = ctypes.c_int32
        return lambda value: lookup(value.encode("utf-8"))
    return None


def runtime_renderer_tokens(run_dir: Path, *, decode_ids=None) -> dict[str, str] | None:
    """Resolve special-token Jinja variables from runtime artifacts when no
    verified serving bundle exists.

    IDs come from weights_manifest.json (or init.json) special_tokens; the
    text comes from the runtime's own embedded tokenizer, never from a
    hardcoded table. Returns None when the values cannot be resolved, so a
    template that requires them still fails loudly under StrictUndefined
    instead of rendering invented delimiters.
    """
    ids = _manifest_special_token_ids(run_dir)
    if not ids:
        return None
    if decode_ids is None:
        decode_ids = _loaded_model_token_decoder(run_dir)
    if decode_ids is None:
        return None
    variables: dict[str, str] = {}
    for name, token_id in sorted(ids.items()):
        try:
            text = decode_ids([token_id])
        except Exception:
            return None
        if not text:
            return None
        variables[name] = text
    return variables or None


def runtime_auto_bos_token(run_dir: Path, *, decode_ids=None) -> str | None:
    """Text of the BOS token the runtime tokenizer adds automatically at encode
    time, or None when the runtime does not auto-add one.

    Templates rendered with a literal leading BOS (HF/GGUF-native templates
    emit ``{{ bos_token }}``) would otherwise be encoded with BOS twice: once
    from the rendered text and once from the encoder's add_bos behavior. The
    text is resolved through the runtime's own tokenizer, never hardcoded.
    """
    special = _manifest_special_tokens(run_dir)
    if not special or not special.get("add_bos_token"):
        return None
    bos_id = special.get("bos_token_id")
    if not isinstance(bos_id, int) or isinstance(bos_id, bool):
        return None
    if decode_ids is None:
        decode_ids = _loaded_model_token_decoder(run_dir)
    if decode_ids is None:
        return None
    try:
        text = decode_ids([bos_id])
    except Exception:
        return None
    return text or None
