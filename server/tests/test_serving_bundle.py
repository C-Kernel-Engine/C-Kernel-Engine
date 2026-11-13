"""Circuit ownership, resolution, and offline serving of immutable assets."""
import hashlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from server.live import _render_with_chat_templates, create_app
from server.runtime import load_manifest_templates, load_tool_protocol
from server.serving_bundle import (
    contract_identity, load_resolved_serving, loaded_serving_identity, resolved_renderer_tokens,
    verify_loaded_libraries, verify_runtime_dependencies,
)
from server.tests.test_native_qwen_jinja_contract import RecordingSession

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from ck_serve_v8 import create_app as create_serving_app
spec = importlib.util.spec_from_file_location("resolve_serving_bundle_v8", ROOT / "version/v8/scripts/resolve_serving_bundle_v8.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def _session_library(path: Path, *, marker: int = 1):
    source = path.with_suffix(".c")
    source.write_text(f"int ck_session_v8_open(void) {{ return {marker}; }}\n")
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(path), str(source)], check=True)
    return ctypes.CDLL(str(path))


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    profiles = source / "serving_profiles"
    profiles.mkdir(parents=True)
    shutil.copyfile(ROOT / "version/v8/serving_profiles/qwen_tools_v1.json", profiles / "qwen_tools_v1.json")
    shutil.copyfile(ROOT / "server/tests/fixtures/qwen_code_xml_compat.jinja", profiles / "test_variant.jinja")
    profile = json.loads((profiles / "qwen_tools_v1.json").read_text())
    # A synthetic profile variant tests generic asset resolution. It is not a
    # shipped Qwen workaround or a serving default.
    profile["variants"]["compat"] = {
        "chat": {"source": "publisher", "variant": "default"},
        "tools": {"source": "profile_asset", "path": "test_variant.jinja"},
        "output_protocol": "qwen_code_xml_raw_v2"}
    (profiles / "qwen_tools_v1.json").write_text(json.dumps(profile))
    circuit = source / "circuit.json"
    circuit.write_text(json.dumps({"name": "test-circuit", "serving": {
        "schema": "cke.circuit_serving.v1", "profile_ref": "serving_profiles/qwen_tools_v1.json", "default_variant": "publisher"}}))
    run = tmp_path / "bundle"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"model": "test-circuit"}))
    (run / "layout_decode.json").write_text(json.dumps({"config": {"context_length": 128}}))
    for name in ("libmodel.so", "libckernel_tokenizer.so", "libckernel_engine.so", "weights.bump", "weights_manifest.map"):
        (run / name).write_bytes(b"test-" + name.encode())
    (run / "chat_template.jinja").write_bytes(b"{% for message in messages %}{{ message.role }}:{{ message.content }}\r\n{% endfor %}assistant:")
    (run / "weights_manifest.json").write_text(json.dumps({"template": json.loads(circuit.read_text())}))
    (run / "model_v8.c").write_text("generated test source")
    maps = source / "kernel_maps"
    maps.mkdir()
    selected_map = {
        "id": "test_kernel", "impl": {"function": "test_kernel"},
    }
    (maps / "test_kernel.json").write_text(json.dumps(selected_map))
    registry = maps / "KERNEL_REGISTRY.json"
    registry.write_text(json.dumps({"kernels": [{**selected_map, "_source_file": "test_kernel.json"}]}))
    for phase in ("prefill", "decode"):
        (run / f"lowered_{phase}_call.json").write_text(json.dumps({
            "operations": [{"function": "test_kernel", "call_abi": {
                "owner": "kernel_map", "kernel_id": "test_kernel", "source_file": "test_kernel.json",
            }}],
        }))
    def identity(path):
        return {"path": str(path), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ir_outputs = {"decode_layout": identity(run / "layout_decode.json"),
                  **{f"{phase}_lowered": identity(run / f"lowered_{phase}_call.json")
                     for phase in ("prefill", "decode")}}
    model = identity(run / "model_v8.c")
    stamps = {
        "ir": {"inputs": {"schema": "ck-v8-ir-bundle-v1", "manifest": identity(run / "weights_manifest.json")}, "outputs": ir_outputs},
        "codegen": {"inputs": {"schema": "ck-v8-codegen-bundle-v1", "artifacts": ir_outputs,
                               "generator_sources": {str(registry): hashlib.sha256(registry.read_bytes()).hexdigest()}},
                    "outputs": {"model_v8.c": model}},
        "runtime": {"inputs": {"schema": "ck-v8-runtime-bundle-v2", "model_source": model}, "outputs": {name: identity(run / name) for name in ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")}}}
    for phase, stamp in stamps.items():
        (run / f".ck_{phase}_bundle.json").write_text(json.dumps(stamp))
    return run, circuit, source


def resolve(setup, variant=None, *, allow_serving_update=False):
    run, circuit, source = setup
    return resolver.resolve_serving_bundle(run, circuit, variant=variant, v8_root=source, allow_serving_update=allow_serving_update)


def _update_test_registry(setup, *, dependencies=None, function=None):
    registry = setup[2] / "kernel_maps/KERNEL_REGISTRY.json"
    data = json.loads(registry.read_text())
    if dependencies is not None:
        data["kernels"][0]["runtime_dependencies"] = dependencies
    if function is not None:
        data["kernels"][0]["impl"]["function"] = function
    registry.write_text(json.dumps(data))
    stamp = setup[0] / ".ck_codegen_bundle.json"
    compiled = json.loads(stamp.read_text())
    compiled["inputs"]["generator_sources"][str(registry)] = hashlib.sha256(registry.read_bytes()).hexdigest()
    stamp.write_text(json.dumps(compiled))


def test_selected_native_dependency_fails_before_first_request(setup, monkeypatch, tmp_path):
    _update_test_registry(setup, dependencies=["sleef_expf16_u10"])
    doc = resolve(setup)
    assert doc["runtime_dependencies"] == ["sleef_expf16_u10"]
    monkeypatch.setenv("CK_SLEEF_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(ValueError, match="Sleef_expf16_u10"):
        verify_runtime_dependencies(load_resolved_serving(setup[0]))
    source = tmp_path / "provider.c"
    source.write_text("float Sleef_expf16_u10(float x) { return x; }\n")
    library = tmp_path / "provider.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    monkeypatch.setenv("CK_SLEEF_LIBRARY", str(library))
    verify_runtime_dependencies(doc)


def test_selected_kernel_dependency_is_bound_to_compiled_call(setup):
    resolve(setup)
    _update_test_registry(setup, function="different_kernel")
    with pytest.raises(ValueError, match="selected kernel map changed"):
        resolve(setup)


def test_changed_registry_requires_rebuild_before_serving(setup):
    resolve(setup)
    registry = setup[2] / "kernel_maps/KERNEL_REGISTRY.json"
    data = json.loads(registry.read_text())
    data["kernels"][0]["runtime_dependencies"] = []
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="kernel registry differs"):
        resolve(setup)


def test_gguf_config_uses_verified_manifest_circuit_identity(setup):
    run, _, _ = setup
    (run / "config.json").write_text(json.dumps({
        "model_type": "test-circuit", "model_name": "Publisher Checkpoint Name",
    }))
    manifest_path = run / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = "test-circuit"
    manifest_path.write_text(json.dumps(manifest))
    ir_stamp = run / ".ck_ir_bundle.json"
    stamp = json.loads(ir_stamp.read_text())
    stamp["inputs"]["manifest"] = {
        "path": str(manifest_path), "size": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    ir_stamp.write_text(json.dumps(stamp))

    assert resolve(setup)["variant"] == "publisher"
    manifest["model"] = "different-circuit"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compiled manifest model"):
        resolve(setup)


def test_explicit_config_model_mismatch_still_fails(setup):
    run, _, _ = setup
    (run / "config.json").write_text(json.dumps({"model": "different-circuit"}))
    with pytest.raises(ValueError, match="runtime configuration"):
        resolve(setup)


def test_gguf_config_without_compiled_model_identity_fails(setup):
    run, _, _ = setup
    (run / "config.json").write_text(json.dumps({
        "model_type": "test-circuit", "model_name": "Publisher Checkpoint Name",
    }))
    with pytest.raises(ValueError, match="compiled manifest model"):
        resolve(setup)


def test_publisher_default_and_explicit_compat(setup):
    native = resolve(setup)
    assert native["variant"] == "publisher"
    assert native["output_protocol"] == "qwen_xml"
    chat, variants, _ = load_manifest_templates(setup[0])
    assert "\r\n" in chat and variants["tool_use"] == chat
    compat = resolve(setup, "compat")
    assert compat["identity"] != native["identity"]
    chat, variants, _ = load_manifest_templates(setup[0])
    assert "\r\n" in chat and "function_calls" in variants["tool_use"]
    assert load_tool_protocol(setup[0], chat, variants) == "qwen_code_xml_raw_v2"
    with pytest.raises(ValueError, match="override conflicts"):
        load_tool_protocol(setup[0], chat + "changed", variants)


def test_resolved_stop_policy_is_bound_to_selected_variant(setup):
    run, _, source = setup
    profile_path = source / "serving_profiles/qwen_tools_v1.json"
    profile = json.loads(profile_path.read_text())
    profile["variants"]["publisher"]["stop_text"] = ["<eos>"]
    profile_path.write_text(json.dumps(profile))
    doc = resolve(setup)
    assert doc["stop_text"] == ["<eos>"]
    assert load_resolved_serving(run)["stop_text"] == ["<eos>"]
    doc["stop_text"] = ["<bad>"]
    doc["identity"] = contract_identity(doc)
    with pytest.raises(ValueError, match="stop policy mismatch"):
        load_resolved_serving(run, document=doc)


def test_publisher_chat_profile_rejects_tools_before_generation(setup):
    run, circuit_path, source = setup
    shutil.copyfile(
        ROOT / "version/v8/serving_profiles/publisher_chat_v1.json",
        source / "serving_profiles/publisher_chat_v1.json",
    )
    circuit = json.loads(circuit_path.read_text())
    circuit["serving"]["profile_ref"] = "serving_profiles/publisher_chat_v1.json"
    circuit_path.write_text(json.dumps(circuit))
    resolved = resolver.resolve_serving_bundle(
        run, circuit_path, v8_root=source, allow_serving_update=True,
    )
    assert resolved["output_protocol"] == "none"
    chat, variants, _ = load_manifest_templates(run)
    assert load_tool_protocol(run, chat, variants) == "none"
    session = RecordingSession(["Hello.", "Hello again."])
    client = TestClient(create_serving_app(
        session, model="test", chat_template=chat, chat_templates=variants,
        tool_protocol="none", viz=False,
    ))
    assert client.post("/v1/responses", json={"model": "test", "input": "Hello."}).status_code == 200
    assert client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "Hello."}],
    }).status_code == 200
    prompts_before_tools = list(session.prompts)
    function = {"name": "read_file", "parameters": {"type": "object", "properties": {}}}
    response_tool = client.post("/v1/responses", json={
        "model": "test", "input": "Read a file", "tools": [{"type": "function", **function}],
    })
    chat_tool = client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "Read a file"}],
        "tools": [{"type": "function", "function": function}],
    })
    assert response_tool.status_code == 501
    assert chat_tool.status_code == 501
    assert session.prompts == prompts_before_tools


def test_muse_circuit_resolves_publisher_atem_and_turn_stops(setup):
    run, circuit_path, source = setup
    muse_circuit = json.loads((ROOT / "version/v8/circuits/muse_glimmer_text.json").read_text())
    declaration = muse_circuit["serving"]
    assert declaration["profile_ref"] == "serving_profiles/muse_glimmer_v1.json"
    shutil.copyfile(
        ROOT / "version/v8" / declaration["profile_ref"],
        source / declaration["profile_ref"],
    )
    circuit = json.loads(circuit_path.read_text())
    circuit["serving"] = declaration
    circuit_path.write_text(json.dumps(circuit))
    (run / "chat_template.jinja").write_bytes(
        (ROOT / "server/tests/fixtures/muse_glimmer_publisher_chat_template.jinja").read_bytes()
    )
    doc = resolver.resolve_serving_bundle(
        run, circuit_path, v8_root=source, allow_serving_update=True,
    )
    assert doc["output_protocol"] == "muse_atem_v1"
    assert doc["stop_text"] == ["<|eot|>", "<|end_of_text|>"]
    assert doc["message_role_map"] == {"developer": "system"}
    assert "<|eom|>" not in doc["stop_text"]
    assert load_resolved_serving(run)["identity"] == doc["identity"]
    # The generic CLI path consumes the resolved bundle, not a model-name rule.
    from server.serving_bundle import resolved_templates
    chat, _, contract = resolved_templates(run, load_resolved_serving(run))
    from server.live import _render_with_chat_templates
    body = type("Body", (), {"instructions": "System policy", "tools": None, "tool_choice": None})()
    rendered = _render_with_chat_templates(
        chat, None, [{"role": "developer", "content": "Developer policy"},
                     {"role": "user", "content": "Hello"}], body, contract,
        renderer_tokens={"bos_token": "<|begin_of_text|>"},
    )
    assert "Developer policy" in rendered
    assert rendered.index("System policy") < rendered.index("Developer policy") < rendered.index("Hello")
    session = RecordingSession(["Hello back."])
    client = TestClient(create_app(
        session, model="test", chat_template=chat,
        chat_contract=contract, renderer_tokens={"bos_token": "<|begin_of_text|>"},
        tool_protocol="muse_atem_v1",
    ))
    response = client.post("/v1/responses", json={
        "model": "test", "instructions": "System policy",
        "input": [{"role": "developer", "content": "Developer policy"},
                  {"role": "user", "content": "Hello"}],
        "max_output_tokens": 16,
    })
    assert response.status_code == 200
    assert "Developer policy" in session.prompts[0]
    tampered = dict(doc, message_role_map={})
    tampered["identity"] = contract_identity(tampered)
    with pytest.raises(ValueError, match="message-role policy mismatch"):
        load_resolved_serving(run, document=tampered)


def test_message_role_policy_changes_identity_without_changing_publisher_jinja(setup):
    run, _, source = setup
    profile_path = source / "serving_profiles/qwen_tools_v1.json"
    before = resolve(setup)
    assert before["message_role_map"] == {}
    chat_hash = before["assets"]["chat"]["sha256"]
    publisher_hash = hashlib.sha256((run / "chat_template.jinja").read_bytes()).hexdigest()

    profile = json.loads(profile_path.read_text())
    profile["variants"]["publisher"]["message_role_map"] = {"developer": "system"}
    profile_path.write_text(json.dumps(profile))
    after = resolve(setup)
    assert after["identity"] != before["identity"]
    assert after["assets"]["chat"]["sha256"] == chat_hash == publisher_hash
    assert (run / "chat_template.jinja").read_bytes() == b"{% for message in messages %}{{ message.role }}:{{ message.content }}\r\n{% endfor %}assistant:"


def test_muse_role_policy_through_normal_bundle_entry_and_streamed_continuation(setup, monkeypatch):
    import server.live as live
    import server.serving_bundle as bundle
    from server.tests.test_muse_http_protocol import CALL, FINAL, TOOL, _Session, _events

    run, circuit_path, source = setup
    declaration = json.loads((ROOT / "version/v8/circuits/muse_glimmer_text.json").read_text())["serving"]
    shutil.copyfile(ROOT / "version/v8" / declaration["profile_ref"], source / declaration["profile_ref"])
    circuit = json.loads(circuit_path.read_text())
    circuit["serving"] = declaration
    circuit_path.write_text(json.dumps(circuit))
    (run / "chat_template.jinja").write_bytes(
        (ROOT / "server/tests/fixtures/muse_glimmer_publisher_chat_template.jinja").read_bytes()
    )
    (run / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
    }))
    resolve(setup, allow_serving_update=True)

    session = _Session([CALL, FINAL])
    session.lib = _session_library(run / "libck_session_v8.so")
    session.close = lambda: None
    monkeypatch.setattr(live, "open_live_session", lambda *_args, **_kwargs: (session, 128, {}))
    # This synthetic bundle uses placeholder model/engine/tokenizer bytes; their
    # loaded-mapping check is exercised separately with real shared libraries.
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_args: None)
    client = TestClient(live.create_live_app_from_run_dir(run, model="muse-test"))
    request = {
        "model": "muse-test", "instructions": "SYSTEM_SENTINEL",
        "input": [{"role": "developer", "content": "DEV_FIRST_SENTINEL"},
                  {"role": "user", "content": "USER_QUERY_SENTINEL"},
                  {"role": "developer", "content": "DEV_SECOND_SENTINEL"}],
        "tools": [TOOL], "stream": True, "max_output_tokens": 16,
    }
    original = json.loads(json.dumps(request))
    first = client.post("/v1/responses", json=request)
    assert first.status_code == 200, first.text
    assert request == original
    events = _events(first)
    assert any(event["type"] == "response.completed" for event in events)
    call = next(event["item"] for event in events
                if event["type"] == "response.output_item.done"
                and event.get("item", {}).get("type") == "function_call")
    response_id = next(event["response"]["id"] for event in events
                       if event["type"] == "response.completed")
    prompt = session.prompts[0]
    for sentinel in ("SYSTEM_SENTINEL", "DEV_FIRST_SENTINEL", "DEV_SECOND_SENTINEL"):
        assert prompt.count(sentinel) == 1
    assert prompt.index("SYSTEM_SENTINEL") < prompt.index("DEV_FIRST_SENTINEL") < prompt.index("USER_QUERY_SENTINEL") < prompt.index("DEV_SECOND_SENTINEL")

    second = client.post("/v1/responses", json={
        "model": "muse-test", "previous_response_id": response_id,
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "cobalt"}],
        "tools": [TOOL], "stream": True, "max_output_tokens": 16,
    })
    assert second.status_code == 200, second.text
    assert any(event["type"] == "response.completed" for event in _events(second))
    assert "cobalt" in session.prompts[1]
    assert session.prompts[1].count("DEV_FIRST_SENTINEL") == 1
    assert session.prompts[1].count("DEV_SECOND_SENTINEL") == 1


def test_invalid_message_role_mapping_fails_bundle_resolution(setup):
    _, _, source = setup
    profile_path = source / "serving_profiles/qwen_tools_v1.json"
    profile = json.loads(profile_path.read_text())
    profile["variants"]["publisher"]["message_role_map"] = {"user": "system"}
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="invalid serving message_role_map"):
        resolve(setup)


def test_publisher_special_tokens_feed_strict_jinja_from_verified_bundle(setup):
    from server.live import TemplateRenderError, _render_with_chat_templates

    run = setup[0]
    (run / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}assistant:")
    (run / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}))
    (run / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": {"content": "<|begin_of_text|>"}}))
    doc = resolve(setup)
    tokens = resolved_renderer_tokens(run, load_resolved_serving(run))
    assert tokens == {"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}
    chat, variants, _ = load_manifest_templates(run)
    with pytest.raises(TemplateRenderError, match="bos_token"):
        _render_with_chat_templates(chat, variants,
                                    [{"role": "user", "content": "Hello."}], None)
    assert _render_with_chat_templates(chat, variants,
                                       [{"role": "user", "content": "Hello."}], None,
                                       renderer_tokens=tokens) == "<|begin_of_text|>user:Hello.assistant:"
    session = RecordingSession(["Hi."])
    client = TestClient(create_app(session, model="test", chat_template=chat,
                                   chat_templates=variants, renderer_tokens=tokens))
    response = client.post("/v1/responses", json={"model": "test", "input": "Hello."})
    assert response.status_code == 200
    assert session.prompts == ["<|begin_of_text|>user:Hello.assistant:"]
    assert "tokenizer_config.json" in doc["assets"]
    (run / "tokenizer_config.json").write_text('{"bos_token":"stale"}')
    with pytest.raises(ValueError, match="stale serving asset"):
        load_resolved_serving(run)


def test_gguf_manifest_ids_bind_publisher_token_text_from_verified_assets(setup):
    from server.live import _render_with_chat_templates

    run = setup[0]
    manifest_path = run / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["special_tokens"] = {"bos_token_id": 2, "eos_token_id": 106}
    manifest_path.write_text(json.dumps(manifest))
    ir_stamp = run / ".ck_ir_bundle.json"
    stamp = json.loads(ir_stamp.read_text())
    stamp["inputs"]["manifest"] = {
        "path": str(manifest_path), "size": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    ir_stamp.write_text(json.dumps(stamp))
    (run / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 2, "content": "<bos>"}, {"id": 106, "content": "<turn|>"},
    ]}))
    (run / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}"
    )
    resolved = resolve(setup)
    tokens = resolved_renderer_tokens(run, load_resolved_serving(run))
    assert tokens == {"bos_token": "<bos>", "eos_token": "<turn|>"}
    # Display decoding may omit control tokens; exact loaded vocabulary lookup
    # verifies their spelling without relying on user-facing decoded text.
    assert resolved_renderer_tokens(
        run, load_resolved_serving(run), decode_ids=lambda _ids: "",
        lookup_token=lambda text: {"<bos>": 2, "<turn|>": 106}.get(text, -1),
    ) == tokens
    with pytest.raises(ValueError, match="loaded tokenizer disagrees"):
        resolved_renderer_tokens(
            run, load_resolved_serving(run), decode_ids=lambda _ids: "",
            lookup_token=lambda text: {"<bos>": 3, "<turn|>": 106}.get(text, -1),
        )
    with pytest.raises(ValueError, match="loaded tokenizer disagrees"):
        resolved_renderer_tokens(run, load_resolved_serving(run),
                                 decode_ids=lambda ids: "<wrong>")
    chat, variants, _ = load_manifest_templates(run)
    assert _render_with_chat_templates(chat, variants, [{"role": "user", "content": "Hello"}],
                                       None, renderer_tokens=tokens) == "<bos>Hello<turn|>"
    (run / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 2, "content": "<different>"}, {"id": 106, "content": "<turn|>"},
    ]}))
    with pytest.raises(ValueError, match="stale serving asset"):
        load_resolved_serving(run)
    assert resolved["assets"]["tokenizer.json"]["sha256"] != hashlib.sha256(
        (run / "tokenizer.json").read_bytes()).hexdigest()


def test_loaded_special_token_lookup_uses_raw_vocabulary_not_display_decode(tmp_path):
    import server.serving_bundle as bundle

    if not hasattr(os, "RTLD_NOLOAD"):
        pytest.skip("loaded-library lookup requires RTLD_NOLOAD")
    source = tmp_path / "model.c"
    source.write_text('''
        #include <stdint.h>
        #include <string.h>
        int32_t ck_model_lookup_token(const char *text) {
            if (!strcmp(text, "<bos>")) return 2;
            if (!strcmp(text, "<eos>")) return 3;
            return -1;
        }
        int ck_model_decode_tokens(const int32_t *ids, int n, char *out, int cap) {
            (void)ids; (void)n; (void)cap; out[0] = 0; return 0;
        }
    ''')
    library_path = tmp_path / "libmodel.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(library_path), str(source)], check=True)
    loaded = ctypes.CDLL(str(library_path))
    assert loaded.ck_model_lookup_token
    assert bundle._loaded_model_token_decoder(tmp_path)([2]) == ""
    lookup = bundle._loaded_model_token_lookup(tmp_path)
    assert lookup is not None
    assert lookup("<bos>") == 2
    assert lookup("<eos>") == 3
    assert lookup("<wrong>") == -1


def test_suppressed_optional_control_token_does_not_block_startup(tmp_path):
    (tmp_path / "weights_manifest.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 1, "eos_token_id": 12},
    }))
    doc = {"assets": {"weights_manifest.json": {"path": "weights_manifest.json"}}}
    tokens = resolved_renderer_tokens(tmp_path, doc, decode_ids=lambda _ids: "")
    assert tokens == {}
    assert _render_with_chat_templates(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        None, [{"role": "user", "content": "Hello"}], None,
        renderer_tokens=tokens,
    ) == "Hello"


def test_gguf_manifest_ids_without_tokenizer_json_use_loaded_decoder(setup, monkeypatch):
    import server.live as live
    import server.serving_bundle as bundle

    run = setup[0]
    manifest_path = run / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["special_tokens"] = {
        "bos_token_id": 2, "eos_token_id": 106, "add_bos_token": True,
    }
    manifest_path.write_text(json.dumps(manifest))
    ir_stamp = run / ".ck_ir_bundle.json"
    stamp = json.loads(ir_stamp.read_text())
    stamp["inputs"]["manifest"] = {
        "path": str(manifest_path), "size": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    ir_stamp.write_text(json.dumps(stamp))
    (run / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"
    )
    resolve(setup)
    doc = load_resolved_serving(run)
    assert "tokenizer.json" not in doc["assets"]
    decoder = lambda ids: {2: "<bos>", 106: "<end_of_turn>"}[ids[0]]
    assert resolved_renderer_tokens(run, doc, decode_ids=decoder) == {
        "bos_token": "<bos>", "eos_token": "<end_of_turn>",
    }

    session = RecordingSession(["ok"])
    session.lib = _session_library(run / "libck_session_v8.so")
    monkeypatch.setattr(live, "open_live_session", lambda *_args, **_kwargs: (session, 128, {}))
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_args: None)
    monkeypatch.setattr(bundle, "_loaded_model_token_decoder", lambda *_args: decoder)
    client = TestClient(live.create_live_app_from_run_dir(run, model="gemma-test"))
    response = client.post("/v1/responses", json={
        "model": "gemma-test", "input": "hello", "max_output_tokens": 1,
    })
    assert response.status_code == 200, response.text
    assert session.prompts == ["hello"]  # Auto-added BOS is not duplicated.

    closed = []
    session.close = lambda: closed.append(True)
    def fail_decode(_ids):
        raise ValueError("decoder failed")

    monkeypatch.setattr(bundle, "_loaded_model_token_decoder", lambda *_args: fail_decode)
    with pytest.raises(ValueError, match="loaded tokenizer cannot decode"):
        live.create_live_app_from_run_dir(run, model="gemma-test")
    assert closed == [True]
    assert session.prompts == ["hello"]  # No generation after failed startup.


def test_gguf_manifest_token_text_conflict_fails_closed(setup):
    run = setup[0]
    (run / "weights_manifest.json").write_text(json.dumps({
        **json.loads((run / "weights_manifest.json").read_text()),
        "special_tokens": {"bos_token_id": 2},
    }))
    manifest_path = run / "weights_manifest.json"
    ir_stamp = run / ".ck_ir_bundle.json"
    stamp = json.loads(ir_stamp.read_text())
    stamp["inputs"]["manifest"] = {
        "path": str(manifest_path), "size": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    ir_stamp.write_text(json.dumps(stamp))
    (run / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 2, "content": "<bos>"}]}))
    (run / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<other>"}))
    resolve(setup)
    with pytest.raises(ValueError, match="conflicting publisher token variable"):
        resolved_renderer_tokens(run, load_resolved_serving(run))


def test_conflicting_publisher_special_tokens_fail_closed(setup):
    run = setup[0]
    (run / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos-a>"}))
    (run / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<bos-b>"}))
    doc = resolve(setup)
    with pytest.raises(ValueError, match="conflicting publisher token variable"):
        resolved_renderer_tokens(run, doc)


def test_publisher_date_helper_has_upstream_format_contract(monkeypatch):
    import server.live as live

    class FixedDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 9, 30)

    monkeypatch.setattr(live, "datetime", FixedDatetime)
    actual = live._render_with_chat_templates(
        "{{ bos_token }}Current date: {{ strftime_now('%Y-%m-%d') }}.",
        None, [{"role": "user", "content": "Hello."}], None,
        renderer_tokens={"bos_token": "<|begin_of_text|>"})
    assert actual == "<|begin_of_text|>Current date: 2026-09-30."


def test_request_reuses_one_clock_snapshot_across_prefix_renders(monkeypatch):
    import server.live as live
    from datetime import datetime

    class RolloverDatetime:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            return (datetime(2026, 9, 30, 23, 59, 59) if cls.calls == 1
                    else datetime(2026, 10, 1, 0, 0, 0))

    monkeypatch.setattr(live, "datetime", RolloverDatetime)
    template = (
        "{{ strftime_now('%Y-%m-%d') }} {{ strftime_now('%Y-%m-%d') }} "
        "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    session = RecordingSession(["Hi."])
    client = TestClient(create_app(session, model="test", chat_template=template))
    response = client.post("/v1/responses", json={"model": "test", "input": "Hello."})
    assert response.status_code == 200
    assert session.prompts == ["2026-09-30 2026-09-30 user:Hello.assistant:"]
    assert RolloverDatetime.calls == 1


def test_copied_bundle_serves_without_profile_or_server_templates(setup):
    resolve(setup, "compat")
    run, _, source = setup
    copied = run.parent / "copied"
    shutil.copytree(run, copied)
    shutil.rmtree(run)
    shutil.rmtree(source)
    chat, variants, _ = load_manifest_templates(copied)
    protocol = load_tool_protocol(copied, chat, variants)
    session = RecordingSession([
        '<function_calls><invoke name="read_file"><parameter name="file_path">/tmp/probe</parameter></invoke></function_calls>',
        "Read complete."])
    client = TestClient(create_app(session, model="test", chat_template=chat, chat_templates=variants, tool_protocol=protocol))
    first = client.post("/v1/responses", json={"model": "test", "input": "Read the probe.", "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]})
    assert first.status_code == 200
    call = next(item for item in first.json()["output"] if item["type"] == "function_call")
    second = client.post("/v1/responses", json={"model": "test", "previous_response_id": first.json()["id"], "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "probe contents"}]})
    assert second.status_code == 200
    assert "Read complete." in str(second.json())


@pytest.mark.parametrize("asset", ["config.json", "layout_decode.json", "weights.bump", "weights_manifest.map", "libmodel.so", "libckernel_tokenizer.so", "libckernel_engine.so"])
def test_stale_runtime_asset_rejected(setup, asset):
    resolve(setup)
    (setup[0] / asset).write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale serving asset"):
        load_manifest_templates(setup[0])


def test_missing_profile_and_unsupported_variant_leave_previous_contract(setup):
    initial = resolve(setup)
    with pytest.raises(ValueError, match="unsupported serving variant"):
        resolve(setup, "unknown")
    assert load_resolved_serving(setup[0])["identity"] == initial["identity"]
    (setup[2] / "serving_profiles/qwen_tools_v1.json").unlink()
    with pytest.raises(ValueError, match="missing or escaping"):
        resolve(setup)
    assert load_resolved_serving(setup[0])["identity"] == initial["identity"]


def test_conflicting_publisher_assets_rejected(setup):
    (setup[0] / "tokenizer_config.json").write_text(json.dumps({"chat_template": "different"}))
    with pytest.raises(ValueError, match="conflicting publisher"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


def test_safetensors_embedded_publisher_template(setup):
    (setup[0] / "chat_template.jinja").unlink()
    (setup[0] / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "default", "template": "{{ messages[0].content }}"}, {"name": "tool_use", "template": "tools={{ tools | tojson }}"}]}))
    doc = resolve(setup)
    assert doc["assets"]["chat"]["source"] == "tokenizer_config.json:default"
    assert doc["assets"]["tools"]["source"] == "tokenizer_config.json:tool_use"


def test_legacy_operator_variant_is_not_publisher_default(setup):
    directory = setup[0] / "additional_chat_templates"
    directory.mkdir()
    (directory / "tool_use.jinja").write_text("operator compatibility override")
    doc = resolve(setup)
    assert doc["assets"]["tools"]["sha256"] == doc["assets"]["chat"]["sha256"]


def test_changed_profile_and_circuit_change_bundle_identity(setup):
    original = resolve(setup)
    circuit = json.loads(setup[1].read_text())
    circuit["serving"]["revision"] = 3
    setup[1].write_text(json.dumps(circuit))
    with pytest.raises(ValueError, match="serving-only changes require"):
        resolve(setup)
    changed_circuit = resolve(setup, allow_serving_update=True)
    assert original["identity"] != changed_circuit["identity"]
    path = setup[2] / "serving_profiles/qwen_tools_v1.json"
    profile = json.loads(path.read_text())
    profile["revision"] = 2
    path.write_text(json.dumps(profile))
    assert resolve(setup, allow_serving_update=True)["identity"] != changed_circuit["identity"]


def test_corrupt_sidecar_does_not_fall_back(setup):
    resolve(setup)
    sidecar = setup[0] / "serving.json"
    doc = json.loads(sidecar.read_text())
    doc["variant"] = "compat"
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_manifest_templates(setup[0])


def test_incompatible_circuit_and_escaped_asset_rejected(setup):
    (setup[0] / "config.json").write_text(json.dumps({"model": "another-circuit"}))
    with pytest.raises(ValueError, match="does not match"):
        resolve(setup)
    (setup[0] / "config.json").write_text(json.dumps({"model": "test-circuit"}))
    (setup[0] / "weights.bump").unlink()
    external = setup[0].parent / "outside.bump"
    external.write_bytes(b"outside")
    (setup[0] / "weights.bump").symlink_to(external)
    with pytest.raises(ValueError, match="escapes bundle"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


def test_profile_reference_cannot_escape_source_root(setup):
    doc = json.loads(setup[1].read_text())
    doc["serving"]["profile_ref"] = "../elsewhere.json"
    setup[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="missing or escaping"):
        resolve(setup)


def test_missing_selected_template_rejects_without_legacy_fallback(setup):
    doc = resolve(setup, "compat")
    (setup[0] / doc["assets"]["tools"]["path"]).unlink()
    with pytest.raises(ValueError, match="invalid resolved serving bundle"):
        load_manifest_templates(setup[0])


def test_loaded_library_identity_rejects_alternate_engine(setup):
    doc = resolve(setup)
    names = ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")
    maps = "\n".join(f"1000-2000 r-xp 00000000 00:01 1 {setup[0] / name}" for name in names)
    verify_loaded_libraries(doc, maps_text=maps)
    elsewhere = setup[0].parent / "elsewhere"
    elsewhere.mkdir()
    alternate = elsewhere / "libckernel_engine.so"
    alternate.write_bytes(b"different engine")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_loaded_libraries(doc, maps_text=maps.replace(str(setup[0] / alternate.name), str(alternate)))
    with pytest.raises(ValueError, match="not loaded"):
        verify_loaded_libraries(doc, maps_text="")


def test_loaded_identity_endpoint_reports_only_verified_bundle_hashes(setup):
    doc = resolve(setup)
    names = ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")
    maps = "\n".join(f"1000-2000 r-xp 00000000 00:01 1 {setup[0] / name}" for name in names)
    verify_loaded_libraries(doc, maps_text=maps)
    session_library = setup[0] / "libck_session_v8.so"
    loaded = _session_library(session_library)
    identity = loaded_serving_identity(
        doc, model="test", session_library=loaded,
    )
    chat, variants, _ = load_manifest_templates(setup[0])
    client = TestClient(create_app(
        RecordingSession(["ok"]), model="test", chat_template=chat, context_length=64,
        chat_templates=variants, loaded_identity=identity,
    ))
    first = client.get("/v1/cke/loaded-identity")
    second = client.get("/v1/cke/loaded-identity")
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["serving_identity"] == doc["identity"]
    assert first.json()["assets_sha256"]["libmodel.so"] == doc["assets"]["libmodel.so"]["sha256"]
    assert first.json()["session_library_sha256"] == hashlib.sha256(session_library.read_bytes()).hexdigest()
    assert first.json()["effective_serving"]["configured_mode"] == "templated"
    assert first.json()["effective_serving"]["active_context_limit"] == 64
    assert first.json()["effective_serving"]["stop_at_eos"] is False
    assert str(setup[0]) not in first.text  # No local bundle paths in HTTP evidence.
    assert client.get("/v1/models").json()["data"][0]["id"] == "test"
    legacy = TestClient(create_app(RecordingSession(["ok"]), model="test", chat_template=chat))
    assert legacy.get("/v1/cke/loaded-identity").status_code == 404
    raw = TestClient(create_app(RecordingSession(["ok"]), model="test",
                                allow_untemplated=True, context_length=64,
                                stop_on_text=("DONE",), loaded_identity=identity))
    raw_identity = raw.get("/v1/cke/loaded-identity").json()
    assert raw_identity["output_protocol"] == doc["output_protocol"]
    assert raw_identity["effective_serving"] == {
        "schema": "cke.effective_serving.v1", "configured_mode": "raw",
        "output_protocol": None, "active_context_limit": 64,
        "default_max_output_tokens": 512, "stop_on_text": ["DONE"],
        "stop_at_eos": False,
    }


def test_loaded_session_library_replacement_cannot_certify_new_path(setup):
    doc = resolve(setup)
    original_path = setup[0] / "libck_session_v8.so"
    loaded = _session_library(original_path, marker=1)
    original_digest = hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert loaded_serving_identity(doc, model="test", session_library=loaded)[
        "session_library_sha256"] == original_digest
    replacement_dir = setup[0] / "replacement"
    replacement_dir.mkdir()
    replacement_path = replacement_dir / "libck_session_v8.so"
    _session_library(replacement_path, marker=2)
    replacement_digest = hashlib.sha256(replacement_path.read_bytes()).hexdigest()
    assert replacement_digest != original_digest
    os.replace(replacement_path, original_path)
    with pytest.raises(ValueError, match="deleted or ambiguous|replaced"):
        loaded_serving_identity(doc, model="test", session_library=loaded)


@pytest.mark.parametrize("allow_serving_update", [False, True])
def test_same_name_computational_change_rejected(setup, allow_serving_update):
    original = resolve(setup)
    circuit = json.loads(setup[1].read_text())
    circuit["operations"] = [{"op": "changed-computation"}]
    setup[1].write_text(json.dumps(circuit))
    with pytest.raises(ValueError, match="circuit differs"):
        resolve(setup, allow_serving_update=allow_serving_update)
    assert load_resolved_serving(setup[0])["identity"] == original["identity"]


@pytest.mark.parametrize("name", ["weights_manifest.json", ".ck_ir_bundle.json", ".ck_codegen_bundle.json", ".ck_runtime_bundle.json", "model_v8.c", "libmodel.so"])
def test_upgrade_requires_unchanged_compilation_provenance(setup, name):
    (setup[0] / name).write_text("{}")
    with pytest.raises((ValueError, KeyError), match="provenance|snapshot"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("loaded_mismatch", [False, True])
def test_chat_and_raw_cli_validate_artifact_integrity(setup, monkeypatch, raw, loaded_mismatch):
    import ck_serve_v8
    import server.serving_bundle as bundle
    resolve(setup)
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    argv = [str(setup[0]), "--no-build", "--no-viz"]
    if raw:
        argv += ["--no-chat-template", "--allow-raw-prompt"]
    if loaded_mismatch:
        closed = []
        class Session:
            def close(self):
                closed.append(True)
        monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: Session())
        monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_: (_ for _ in ()).throw(ValueError("loaded library identity mismatch")))
        with pytest.raises(ValueError, match="loaded library identity mismatch"):
            ck_serve_v8.main(argv)
        assert closed == [True]
    else:
        (setup[0] / "libmodel.so").write_bytes(b"stale")
        monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: pytest.fail("opened stale session"))
        with pytest.raises(ValueError, match="stale serving asset"):
            ck_serve_v8.main(argv)


def test_cli_exposes_loaded_identity_after_runtime_verification(setup, monkeypatch):
    import ck_serve_v8
    import server.serving_bundle as bundle
    import uvicorn

    doc = resolve(setup)
    library = setup[0] / "libck_session_v8.so"
    loaded = _session_library(library)
    closed = []
    class Session:
        lib = loaded
        def close(self):
            closed.append(True)
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: Session())
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_: None)
    served = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_kwargs: served.append(app))

    assert ck_serve_v8.main([str(setup[0]), "--no-build", "--no-viz", "--model-name", "test-model"]) == 0
    assert closed == [True]
    response = TestClient(served[0]).get("/v1/cke/loaded-identity")
    assert response.status_code == 200
    assert response.json()["serving_identity"] == doc["identity"]
    assert response.json()["model"] == "test-model"


def test_cli_rejects_missing_selected_dependency_before_listening(setup, monkeypatch):
    import ck_serve_v8
    import server.serving_bundle as bundle
    import uvicorn

    _update_test_registry(setup, dependencies=["sleef_expf16_u10"])
    resolve(setup)
    closed = []
    class Session:
        lib = object()
        def close(self):
            closed.append(True)
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: Session())
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_: None)
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: pytest.fail("served without dependency"))
    monkeypatch.setenv("CK_SLEEF_LIBRARY", str(setup[0] / "missing.so"))
    with pytest.raises(ValueError, match="Sleef_expf16_u10"):
        ck_serve_v8.main([str(setup[0]), "--no-build", "--no-viz"])
    assert closed == [True]


def test_raw_build_without_chat_assets_and_normal_chat_rejection(setup, monkeypatch):
    from types import SimpleNamespace
    import ck_run_v8
    import ck_serve_v8
    (setup[0] / "chat_template.jinja").unlink()
    raw = SimpleNamespace(resolve_serving=False, serving_variant=None, no_chat_template=True)
    # This is the same post-compile step used by raw/token-ID build pipelines.
    ck_run_v8.step_resolve_serving(raw, setup[0], setup[0] / "weights_manifest.json")
    assert not (setup[0] / "serving.json").exists()
    chat = SimpleNamespace(resolve_serving=True, serving_variant=None, no_chat_template=False)
    canonical = setup[2] / "circuits"
    canonical.mkdir()
    shutil.copyfile(setup[1], canonical / "test-circuit.json")
    monkeypatch.setattr(ck_run_v8, "V8_ROOT", setup[2])
    monkeypatch.setattr(__import__("resolve_serving_bundle_v8"), "resolve_serving_bundle", lambda run, circuit, **kw: resolver.resolve_serving_bundle(run, circuit, v8_root=setup[2], **kw))
    with pytest.raises(ValueError, match="publisher default template is missing"):
        ck_run_v8.step_resolve_serving(chat, setup[0], setup[0] / "weights_manifest.json")
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: pytest.fail("opened without chat assets"))
    with pytest.raises(ValueError, match="normal chat serving requires"):
        ck_serve_v8.main([str(setup[0]), "--no-build", "--no-viz"])


def test_runtime_renderer_tokens_from_manifest_ids_without_serving_bundle(tmp_path):
    """No serving.json: variables resolve from manifest IDs via the runtime tokenizer."""
    from server.serving_bundle import runtime_renderer_tokens

    (tmp_path / "weights_manifest.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 2, "eos_token_id": 106, "pad_token_id": 0},
    }))
    pieces = {2: "<bos>", 106: "<end_of_turn>", 0: "<pad>"}
    tokens = runtime_renderer_tokens(tmp_path, decode_ids=lambda ids: pieces[ids[0]])
    assert tokens == {"bos_token": "<bos>", "eos_token": "<end_of_turn>", "pad_token": "<pad>"}


def test_runtime_renderer_tokens_init_json_fallback(tmp_path):
    from server.serving_bundle import runtime_renderer_tokens

    (tmp_path / "init.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 2},
    }))
    tokens = runtime_renderer_tokens(tmp_path, decode_ids=lambda ids: "<bos>")
    assert tokens == {"bos_token": "<bos>"}


def test_runtime_renderer_tokens_stays_none_when_unresolvable(tmp_path):
    from server.serving_bundle import runtime_renderer_tokens

    # No artifacts at all: None, so StrictUndefined still fails loudly.
    assert runtime_renderer_tokens(tmp_path, decode_ids=lambda ids: "<x>") is None
    (tmp_path / "weights_manifest.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 2},
    }))
    # Empty decode text is not a usable delimiter.
    assert runtime_renderer_tokens(tmp_path, decode_ids=lambda ids: "") is None
    # Decode failure is not silently ignored either.
    def boom(ids):
        raise RuntimeError("no tokenizer")
    assert runtime_renderer_tokens(tmp_path, decode_ids=boom) is None


@pytest.mark.parametrize("stream", [False, True])
def test_flash_next_declared_publisher_tools_through_bundle_entry(setup, monkeypatch, stream):
    """Real resolver/renderer/HTTP parser, scripted native output: not model certification."""
    import server.live as live
    import server.serving_bundle as bundle
    from server.tests.test_native_qwen_jinja_contract import TOOL, FIXTURE, FIXTURE_SHA256
    from server.tests.test_muse_http_protocol import _events

    run, circuit_path, source = setup
    declaration = json.loads((ROOT / "version/v8/circuits/qwen4_exp.json").read_text())["serving"]
    assert declaration == {
        "schema": "cke.circuit_serving.v1",
        "profile_ref": "serving_profiles/qwen_tools_v1.json",
        "default_variant": "publisher",
    }
    circuit = json.loads(circuit_path.read_text())
    circuit["serving"] = declaration
    circuit_path.write_text(json.dumps(circuit))
    (run / "chat_template.jinja").unlink()
    with pytest.raises(ValueError, match="publisher default template is missing"):
        resolve(setup, allow_serving_update=True)
    publisher = FIXTURE.read_bytes()
    assert hashlib.sha256(publisher).hexdigest() == FIXTURE_SHA256
    (run / "chat_template.jinja").write_bytes(publisher)
    resolved = resolve(setup, allow_serving_update=True)
    assert resolved["output_protocol"] == "qwen_xml"
    assert resolved["assets"]["chat"]["sha256"] == FIXTURE_SHA256
    assert (run / "chat_template.jinja").read_bytes() == publisher

    call_text = '<tool_call>\n<function=read_file>\n<parameter=file_path>\n/tmp/flash-read.txt\n</parameter>\n</function>\n</tool_call>'
    session = RecordingSession([call_text, "FLASH_OK_837"])
    session.lib = _session_library(run / "libck_session_v8.so")
    monkeypatch.setattr(live, "open_live_session", lambda *_a, **_k: (session, 8192, {}))
    # Placeholder native libraries are not an artifact or numerical acceptance.
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_a: None)
    client = TestClient(live.create_live_app_from_run_dir(run, model="generic-test-name"))

    def completed(response):
        assert response.status_code == 200, response.text
        if not stream:
            return response.json()
        events = _events(response)
        terminal = [e for e in events if e["type"] == "response.completed"]
        assert len(terminal) == 1
        assert not any(e["type"] in ("error", "response.failed") for e in events)
        return terminal[0]["response"]

    first = completed(client.post("/v1/responses", json={
        "model": "generic-test-name", "input": "Read /tmp/flash-read.txt",
        "tools": [TOOL], "stream": stream, "max_output_tokens": 128,
    }))
    calls = [item for item in first["output"] if item["type"] == "function_call"]
    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "read_file" and call["call_id"]
    assert json.loads(call["arguments"]) == {"file_path": "/tmp/flash-read.txt"}
    assert "<tools>" in session.prompts[0] and "read_file" in session.prompts[0]
    invalid = client.post("/v1/responses", json={
        "model": "generic-test-name", "previous_response_id": first["id"],
        "input": [{"type": "function_call_output", "call_id": "wrong-call-id", "output": "INVALID"}],
        "tools": [TOOL], "stream": False,
    })
    assert invalid.status_code == 400
    assert len(session.prompts) == 1
    final = completed(client.post("/v1/responses", json={
        "model": "generic-test-name", "previous_response_id": first["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "FLASH_OK_837"}],
        "tools": [TOOL], "stream": stream, "max_output_tokens": 128,
    }))
    assert final["status"] == "completed"
    assert final["output_text"] == "FLASH_OK_837"
    prompt = session.prompts[1]
    assert prompt.count("FLASH_OK_837") == 1 and "INVALID" not in prompt
    assert prompt.index("Read /tmp/flash-read.txt") < prompt.index("<function=read_file>") < prompt.index("<tool_response>") < prompt.index("FLASH_OK_837")
