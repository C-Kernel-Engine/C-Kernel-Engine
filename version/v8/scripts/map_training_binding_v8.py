"""Resolve direct forward/VJP bindings declared by kernel-map semantics.

This bounded contract covers maps whose training forward uses the same provider
and saves existing operands. Cached-forward rewrites remain separate rules.
"""

from copy import deepcopy
import re


def resolve_training_binding(kernel_map, rules, *, inputs, weights, output, constants):
    binding = kernel_map.get("training_contract", {}).get("differentiation")
    if binding is None:
        return None
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "schema",
            "operation_interface",
            "gradient_rule",
            "saved_operands",
            "constant_bindings",
        }
        or binding["schema"] != "cke.training.direct-vjp.v1"
        or not kernel_map.get("operation_interface")
        or binding["operation_interface"] != kernel_map["operation_interface"]
    ):
        raise ValueError("invalid map-owned differentiation contract")
    if set(inputs) != {p["name"] for p in kernel_map.get("inputs", [])} or set(
        weights
    ) != {p["name"] for p in kernel_map.get("weights", [])}:
        raise ValueError("training operands disagree with provider ports")
    for ports, refs in (
        (kernel_map.get("inputs", []), inputs),
        (kernel_map.get("weights", []), weights),
    ):
        for port in ports:
            if port.get("dtype") != "fp32" or refs[port["name"]].get("dtype") != "fp32":
                raise ValueError("direct VJP requires explicit FP32 operands")
    outputs = kernel_map.get("outputs", [])
    if (
        len(outputs) != 1
        or outputs[0].get("dtype") != "fp32"
        or output.get("dtype") != "fp32"
    ):
        raise ValueError("direct VJP requires one FP32 output")
    rule_name = binding["gradient_rule"]
    if not isinstance(rule_name, str):
        raise ValueError("gradient rule must be an explicit identity")
    rule = rules.get(rule_name)
    if not isinstance(rule, dict) or rule.get("status") != "ready":
        raise ValueError("map names an unavailable gradient rule")
    saved = {}
    if not isinstance(binding["saved_operands"], dict):
        raise ValueError("invalid saved operand bindings")
    for name, source in binding["saved_operands"].items():
        if not isinstance(source, str) or source.count(":") != 1:
            raise ValueError("invalid saved operand reference")
        kind, port = source.split(":")
        refs = {"input": inputs, "weight": weights}.get(kind)
        if refs is None or port not in refs:
            raise ValueError("saved operand is not a declared port")
        saved[name] = deepcopy(refs[port])
    if set(saved) != set(rule.get("requires_saved", [])) | set(
        rule.get("extra_saved", [])
    ):
        raise ValueError("saved operands disagree with gradient rule")
    result = deepcopy(constants)
    aliases = binding["constant_bindings"]
    if not isinstance(aliases, dict):
        raise ValueError("invalid training constant bindings")
    for name, source in aliases.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError("invalid training constant name")
        if isinstance(source, str):
            if source not in constants:
                raise ValueError("missing forward dimension for training")
            value = constants[source]
        else:
            value = source
        if type(value) is not int or not 0 < value < 2**63:
            raise ValueError("training dimension must be a positive bounded integer")
        result[name] = value
    return rule_name, result, saved
