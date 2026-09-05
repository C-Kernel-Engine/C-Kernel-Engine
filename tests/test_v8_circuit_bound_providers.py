#!/usr/bin/env python3
"""Fail-closed provider bindings for circuit-bound MLA ops.

The compiler keeps no default provider for the ops in
``CIRCUIT_BOUND_PROVIDER_OPS``: every circuit that uses them must declare an
exact provider under its ``kernels`` bindings, and the binding must name a
registered kernel. These tests prove the resolver fails closed when the
binding is missing or unknown, and that the in-tree circuits resolve.
"""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
CIRCUITS_DIR = ROOT / "version" / "v8" / "circuits"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_binding_tests", V8_BUILD_PATH)
kimi_tests = _load_module(
    "kimi_template_tests_for_bindings", ROOT / "tests" / "test_v8_kimi_template.py"
)


def _tiny_kimi_manifest() -> dict:
    return kimi_tests._make_tiny_kimi_manifest()


@contextlib.contextmanager
def _binding_overridden(manifest: dict, op_name: str, value):
    """Remove or replace a binding in every template source hydration merges.

    ``_hydrate_manifest_template`` merges the built-in circuit doc with the
    manifest-embedded template, so a binding must be stripped from both to
    simulate an out-of-tree circuit that never declared it.
    """
    kernels = manifest["template"].setdefault("kernels", {})
    sentinel = object()
    saved = kernels.get(op_name, sentinel)
    if value is None:
        kernels.pop(op_name, None)
    else:
        kernels[op_name] = value

    original_loader = build_ir_v8._load_builtin_template_doc

    def patched_loader(name, *args, **kwargs):
        doc = original_loader(name, *args, **kwargs)
        if isinstance(doc, dict) and str(doc.get("name", "")) == "kimi_vl":
            doc = copy.deepcopy(doc)
            doc_kernels = doc.get("kernels")
            if isinstance(doc_kernels, dict):
                if value is None:
                    doc_kernels.pop(op_name, None)
                else:
                    doc_kernels[op_name] = value
        return doc

    build_ir_v8._load_builtin_template_doc = patched_loader
    try:
        yield
    finally:
        build_ir_v8._load_builtin_template_doc = original_loader
        if saved is sentinel:
            kernels.pop(op_name, None)
        else:
            kernels[op_name] = saved


def _lower_decode_ops(manifest: dict) -> list:
    ops = build_ir_v8.build_ir1_direct(
        manifest, ROOT / "tests" / "binding.synthetic.json", mode="decode"
    )
    registry = build_ir_v8.load_kernel_registry()
    return build_ir_v8.generate_ir_lower_1(ops, registry, manifest, "decode")


@contextlib.contextmanager
def _registry_with_added_providers(providers: list):
    """Register extra test-double providers during a resolution run."""
    original_loader = build_ir_v8.load_kernel_registry

    def patched_loader():
        registry = copy.deepcopy(original_loader())
        registry.setdefault("kernels", []).extend(copy.deepcopy(providers))
        return registry

    build_ir_v8.load_kernel_registry = patched_loader
    try:
        yield
    finally:
        build_ir_v8.load_kernel_registry = original_loader


class CircuitBoundProviderTests(unittest.TestCase):
    def test_in_tree_circuits_declare_every_bound_provider(self) -> None:
        registry = build_ir_v8.load_kernel_registry()
        registry_ids = {k["id"] for k in registry.get("kernels", [])}
        for name in ("kimi_vl", "instella_moe"):
            with self.subTest(circuit=name):
                doc = json.loads(
                    (CIRCUITS_DIR / f"{name}.json").read_text(encoding="utf-8")
                )
                bindings = doc.get("kernels", {})
                for op in build_ir_v8.CIRCUIT_BOUND_PROVIDER_OPS:
                    target = str(bindings.get(op, "") or "").strip()
                    self.assertTrue(
                        target,
                        f"{name} must declare \"{op}\" in its kernels bindings",
                    )
                    self.assertIn(target, registry_ids)

    def test_in_tree_circuit_resolves_bound_providers(self) -> None:
        ops = _lower_decode_ops(_tiny_kimi_manifest())
        by_op = {}
        for op in ops:
            by_op.setdefault(str(op.get("op", "")), str(op.get("kernel", "")))
        self.assertEqual(
            by_op["partial_rope_concat"],
            "deepseek_mla_partial_rope_concat_packed_f32",
        )
        self.assertEqual(by_op["kv_a_layernorm"], "rmsnorm_forward_strided_f32")
        self.assertEqual(
            by_op["kv_lora_decompress"], "deepseek_mla_kv_decompress_bf16"
        )
        self.assertEqual(
            by_op["mla_kv_cache_store"], "deepseek_mla_kv_cache_store_f32"
        )

    def test_missing_binding_fails_for_sequence_ops(self) -> None:
        for op_name in ("partial_rope_concat", "kv_a_layernorm", "kv_lora_decompress"):
            with self.subTest(op=op_name):
                manifest = _tiny_kimi_manifest()
                with _binding_overridden(manifest, op_name, None):
                    with self.assertRaises(RuntimeError) as ctx:
                        build_ir_v8.build_ir1_direct(
                            manifest,
                            ROOT / "tests" / "binding.synthetic.json",
                            mode="decode",
                        )
                message = str(ctx.exception)
                self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
                self.assertIn(op_name, message)
                self.assertIn("kimi_vl", message)
                self.assertIn("kernels", message)

    def test_missing_binding_fails_for_auto_inserted_store_ops(self) -> None:
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "mla_kv_cache_store", None):
            with self.assertRaises(RuntimeError) as ctx:
                _lower_decode_ops(manifest)
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("mla_kv_cache_store", message)
        self.assertIn("kimi_vl", message)
        self.assertIn("kernels", message)

    def test_missing_binding_fails_for_auto_inserted_batch_store_op(self) -> None:
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "mla_kv_cache_batch_store", None):
            ops = build_ir_v8.build_ir1_direct(
                manifest, ROOT / "tests" / "binding.synthetic.json", mode="prefill"
            )
            registry = build_ir_v8.load_kernel_registry()
            with self.assertRaises(RuntimeError) as ctx:
                build_ir_v8.generate_ir_lower_1(ops, registry, manifest, "prefill")
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("mla_kv_cache_batch_store", message)

    def test_unknown_binding_target_fails_for_sequence_ops(self) -> None:
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "partial_rope_concat", "no_such_provider_f32"):
            with self.assertRaises(RuntimeError) as ctx:
                build_ir_v8.build_ir1_direct(
                    manifest, ROOT / "tests" / "binding.synthetic.json", mode="decode"
                )
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("partial_rope_concat", message)
        self.assertIn("no_such_provider_f32", message)

    def test_unknown_binding_target_fails_for_store_ops(self) -> None:
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "mla_kv_cache_store", "no_such_provider_f32"):
            with self.assertRaises(RuntimeError) as ctx:
                _lower_decode_ops(manifest)
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("mla_kv_cache_store", message)
        self.assertIn("no_such_provider_f32", message)

    def test_wrong_class_binding_fails_for_sequence_ops(self) -> None:
        # memcpy is a registered provider, but it implements residual_save,
        # not partial_rope_concat.
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "partial_rope_concat", "memcpy"):
            with self.assertRaises(RuntimeError) as ctx:
                build_ir_v8.build_ir1_direct(
                    manifest, ROOT / "tests" / "binding.synthetic.json", mode="decode"
                )
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("partial_rope_concat", message)
        self.assertIn("memcpy", message)
        self.assertIn("wrong operation class", message)
        self.assertIn("expected='partial_rope_concat'", message)
        self.assertIn("actual='residual_save'", message)

    def test_wrong_class_binding_fails_for_auto_inserted_store_op(self) -> None:
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(manifest, "mla_kv_cache_store", "memcpy"):
            with self.assertRaises(RuntimeError) as ctx:
                _lower_decode_ops(manifest)
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("mla_kv_cache_store", message)
        self.assertIn("memcpy", message)
        self.assertIn("wrong operation class", message)
        self.assertIn("expected='mla_kv_cache_store'", message)
        self.assertIn("actual='residual_save'", message)

    def test_wrong_phase_binding_fails_for_sequence_ops(self) -> None:
        double = {
            "id": "test_prefill_only_rope_concat_f32",
            "op": "partial_rope_concat",
            "selection": {"status": "production", "phases": ["prefill"]},
            "impl": {"function": "test_prefill_only_rope_concat_f32"},
        }
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(
            manifest, "partial_rope_concat", "test_prefill_only_rope_concat_f32"
        ):
            with _registry_with_added_providers([double]):
                with self.assertRaises(RuntimeError) as ctx:
                    build_ir_v8.build_ir1_direct(
                        manifest,
                        ROOT / "tests" / "binding.synthetic.json",
                        mode="decode",
                    )
        message = str(ctx.exception)
        self.assertIn("HARD KERNEL RESOLUTION FAULT", message)
        self.assertIn("partial_rope_concat", message)
        self.assertIn("test_prefill_only_rope_concat_f32", message)
        self.assertIn("does not support the active phase", message)
        self.assertIn("phase='decode'", message)

    def test_store_function_name_comes_from_kernel_map(self) -> None:
        # The provider ID deliberately differs from impl.function; the
        # inserted store op must take its C function name from the map.
        double = {
            "id": "test_mla_store_provider_alias",
            "op": "mla_kv_cache_store",
            "selection": {"status": "production", "phases": ["prefill", "decode"]},
            "impl": {"function": "test_mla_store_c_impl_fn"},
        }
        manifest = _tiny_kimi_manifest()
        with _binding_overridden(
            manifest, "mla_kv_cache_store", "test_mla_store_provider_alias"
        ):
            with _registry_with_added_providers([double]):
                ops = _lower_decode_ops(manifest)
        stores = [op for op in ops if op.get("op") == "mla_kv_cache_store"]
        self.assertTrue(stores, "expected auto-inserted mla_kv_cache_store ops")
        for op in stores:
            self.assertEqual(op.get("kernel"), "test_mla_store_provider_alias")
            self.assertEqual(op.get("function"), "test_mla_store_c_impl_fn")


if __name__ == "__main__":
    unittest.main()
