"""Fail-closed structural checks for generated component boundaries.

A structural PASS is not numerical parity or runtime certification. Hosts may
prepare inputs and call steps, but an internal edge must be produced and consumed
inside the same lowered graph. No model names or application categories occur here.
"""

from copy import deepcopy


def audit_generated_composition(ir, boundaries, output, codegen_report):
    """Check explicit producer/consumer boundaries against actual forward IR.

    Each boundary has `tensor`, `producer` and `consumer` operation names.
    Checking only a common library filename would accept a host-stitched pipeline.
    """
    errors = []
    if not isinstance(codegen_report, dict) or codegen_report.get("skipped_ops") != []:
        errors.append("missing or incomplete code generation")
    ops = [op for op in ir.get("ops", []) if op.get("phase") == "forward"]
    by_name = {}
    producers = {}
    dependencies = {}
    for index, op in enumerate(ops):
        name = op.get("op")
        if not isinstance(name, str) or name in by_name:
            errors.append("missing or duplicate operation identity")
        by_name[name] = (index, op)
        refs = op.get("dataflow", {})
        read = {v.get("tensor") for v in refs.get("inputs", {}).values()}
        read |= {v.get("tensor") for v in op.get("weights", {}).values()}
        for value in refs.get("outputs", {}).values():
            tensor = value.get("tensor")
            if not tensor or tensor in producers:
                errors.append("missing or duplicate tensor producer")
            producers[tensor] = name
            dependencies[tensor] = read
    if not boundaries:
        errors.append("no declared component boundaries")
    ancestors = set()
    pending = [output]
    while pending:
        tensor = pending.pop()
        if tensor in ancestors:
            continue
        ancestors.add(tensor)
        pending.extend(dependencies.get(tensor, ()))
    if output not in producers:
        errors.append("output is not generated")
    for edge in boundaries:
        tensor, producer, consumer = (
            edge.get(k) for k in ("tensor", "producer", "consumer")
        )
        if (
            producers.get(tensor) != producer
            or producer not in by_name
            or consumer not in by_name
        ):
            errors.append("boundary lacks its declared generated producer/consumer")
            continue
        pi, pop = by_name[producer]
        ci, cop = by_name[consumer]
        writes = [
            v for v in pop["dataflow"]["outputs"].values() if v.get("tensor") == tensor
        ]
        reads = [
            v
            for v in cop.get("dataflow", {}).get("inputs", {}).values()
            if v.get("tensor") == tensor
        ]
        metadata = ir.get("tensors", {}).get(tensor, {})
        if pi >= ci or not reads or metadata.get("kind") not in ("activation", "aux"):
            errors.append("boundary is not an ordered internal activation edge")
        if tensor not in ancestors:
            errors.append("boundary is disconnected from final output")
        for ref in writes + reads:
            if any(ref.get(k) != metadata.get(k) for k in ("shape", "dtype", "numel")):
                errors.append("boundary shape/dtype/extent differs")
    return dict(
        schema="cke.generated-composition-audit.v1",
        structural="FAIL" if errors else "PASS",
        errors=errors,
        execution="NOT_TESTED",
        numerical_parity="NOT_TESTED",
    )


def component_only_status(mode):
    if mode not in ("inference", "training"):
        raise ValueError("unsupported execution mode")
    return dict(
        schema="cke.generated-composition-status.v1",
        mode=mode,
        structural="NOT_IMPLEMENTED",
        execution="NOT_TESTED",
        end_to_end="NOT_TESTED",
        host_component_handoff=True,
        reason="Separate generated components with host-owned feature handoff; no composed generated entry.",
    )


def require_generated_e2e(
    report, *, ir=None, boundaries=None, output=None, codegen_report=None
):
    """Reject a completion claim without composed execution AND independent proof.

    Callers must retain individual verdicts; component execution cannot substitute
    for this gate. Structural audit alone cannot establish execution or parity.
    """
    status = report.get("generated_composition", {})
    # Status labels alone cannot certify a connected graph. Recompute structure
    # from compiler artifacts supplied by the certification caller.
    audit = audit_generated_composition(
        ir or {}, boundaries or [], output, codegen_report
    )
    if audit["structural"] != "PASS":
        raise ValueError(
            "generated E2E lacks verified composition: " + "; ".join(audit["errors"])
        )
    required = ("structural", "execution", "numerical_parity", "export_reload")
    missing = [k for k in required if status.get(k) != "PASS"]
    if status.get("mode") == "training":
        missing.extend(
            k
            for k in ("gradients", "optimizer", "frozen_parameters")
            if status.get(k) != "PASS"
        )
    if status.get("mode") not in ("inference", "training"):
        missing.append("mode")
    if status.get("host_component_handoff") is not False:
        missing.append("host_component_handoff")
    if missing:
        raise ValueError("generated E2E not certified: " + ", ".join(missing))
    return deepcopy(status)
