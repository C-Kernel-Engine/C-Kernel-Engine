"""Typed borrowed entry arguments for declared native component calls."""

import re


def resolve_entry_argument(entry, kernel, argument):
    """Return a parameter identifier only when both ABI declarations agree."""
    source = argument.get("source", "")
    name = source.removeprefix("entry:")
    if not source.startswith("entry:") or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", name
    ):
        raise ValueError("invalid entry argument identifier")
    params = entry.get("params", []) if isinstance(entry, dict) else []
    matches = [p for p in params if p.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            "entry argument must resolve to exactly one declared parameter"
        )
    declaration = kernel.get("impl", {}).get("c_declaration", "")
    signature = re.search(r"\((.*)\)\s*;\s*$", declaration)
    parts = signature.group(1).split(",") if signature else []
    abi = kernel.get("call_abi", {}).get("params", [])
    indices = [i for i, p in enumerate(abi) if p == argument]
    if len(parts) != len(abi) or len(indices) != 1 or argument.get("cast"):
        raise ValueError("entry argument requires an uncast canonical C declaration")
    declared = re.sub(
        r"\b" + re.escape(argument["name"]) + r"\s*$", "", parts[indices[0]]
    ).strip()
    normalize = lambda value: re.sub(r"\s+", "", value)
    if normalize(declared) != normalize(matches[0].get("c_type", "")):
        raise ValueError("entry argument type disagrees with provider declaration")
    return name
