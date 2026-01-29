#!/usr/bin/env python3
"""
memory_planner_v6_6.py - Assign physical buffers based on IR1 dataflow graph.

This replaces the buggy ping-pong buffer logic with explicit dataflow-based assignment.

PIPELINE POSITION:
    IR1 (with dataflow) → Kernel Resolution → MEMORY PLANNER → IR Lower

INPUT:
    - IR1 ops with dataflow info (from build_ir_v6_6.py)
    - Kernel maps (to know dtype requirements)

OUTPUT:
    - Buffer assignments per op: {op_id: {input_name: buffer, output_name: buffer}}

PHYSICAL BUFFERS:
    - A_EMBEDDED_INPUT  : Main activation buffer 1 (FP32)
    - A_LAYER_INPUT     : Main activation buffer 2 (FP32/Q8)
    - A_RESIDUAL        : Saved residual for skip connections (FP32)
    - A_ATTN_SCRATCH    : Q/K/V projections and attention output (FP32)
    - A_MLP_SCRATCH     : MLP gate_up and swiglu output (FP32)
    - A_KV_CACHE        : KV cache (persistent across tokens)
    - A_LOGITS          : Final logits output (FP32)
    - A_LAYER_OUTPUT    : Layer output buffer (FP32)

STRATEGY:
    1. Map logical slots to physical buffers
    2. Track which buffer contains which data after each op
    3. For ping-pong between main stream ops, alternate buffers
    4. Output explicit assignments that IR Lower can use directly
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL BUFFER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicalBuffer:
    """A physical memory buffer."""
    name: str           # C macro name (e.g., "A_LAYER_INPUT")
    dtype: str          # Current data type in buffer
    last_writer: int    # op_id that last wrote to this buffer
    can_hold: List[str] # Data types this buffer can hold


# Available physical buffers
PHYSICAL_BUFFERS = {
    "A_EMBEDDED_INPUT": PhysicalBuffer(
        name="A_EMBEDDED_INPUT",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32", "q8_0", "q8_k"]
    ),
    "A_LAYER_INPUT": PhysicalBuffer(
        name="A_LAYER_INPUT",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32", "q8_0", "q8_k"]
    ),
    "A_RESIDUAL": PhysicalBuffer(
        name="A_RESIDUAL",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32"]
    ),
    "A_ATTN_SCRATCH": PhysicalBuffer(
        name="A_ATTN_SCRATCH",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32"]
    ),
    "A_MLP_SCRATCH": PhysicalBuffer(
        name="A_MLP_SCRATCH",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32"]
    ),
    "A_LAYER_OUTPUT": PhysicalBuffer(
        name="A_LAYER_OUTPUT",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32"]
    ),
    "A_LOGITS": PhysicalBuffer(
        name="A_LOGITS",
        dtype="fp32",
        last_writer=-1,
        can_hold=["fp32"]
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SLOT TO BUFFER MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Initial mapping of logical slots to physical buffers
# This is the "default" assignment - the planner may override for specific cases
SLOT_TO_BUFFER_DEFAULT = {
    # Main activation stream (ping-pongs between two buffers)
    "main_stream": "A_EMBEDDED_INPUT",
    "main_stream_q8": "A_LAYER_INPUT",  # Quantized stream uses other buffer

    # Residual storage
    "residual": "A_RESIDUAL",

    # Attention scratch
    "q_scratch": "A_ATTN_SCRATCH",
    "k_scratch": "A_ATTN_SCRATCH",  # K/V share with Q (sequential access)
    "v_scratch": "A_ATTN_SCRATCH",
    "attn_scratch": "A_ATTN_SCRATCH",

    # MLP scratch
    "mlp_scratch": "A_MLP_SCRATCH",

    # KV cache (handled specially)
    "kv_cache": "kv_cache",  # Not a bump buffer

    # Output
    "logits": "A_LOGITS",
}


@dataclass
class BufferState:
    """Tracks the current state of all physical buffers."""
    buffers: Dict[str, PhysicalBuffer] = field(default_factory=dict)

    # Track which buffer currently holds data for each logical purpose
    main_stream_buffer: str = "A_EMBEDDED_INPUT"  # Alternates
    main_stream_q8_buffer: str = "A_LAYER_INPUT"  # Alternates opposite

    def __post_init__(self):
        # Initialize buffer state
        for name, buf in PHYSICAL_BUFFERS.items():
            self.buffers[name] = PhysicalBuffer(
                name=buf.name,
                dtype=buf.dtype,
                last_writer=buf.last_writer,
                can_hold=buf.can_hold.copy()
            )

    def swap_main_stream(self):
        """Swap the main stream buffers (ping-pong)."""
        self.main_stream_buffer, self.main_stream_q8_buffer = \
            self.main_stream_q8_buffer, self.main_stream_buffer

    def get_buffer_for_slot(self, slot: str) -> str:
        """Get the current physical buffer for a logical slot."""
        if slot == "main_stream":
            return self.main_stream_buffer
        elif slot == "main_stream_q8":
            return self.main_stream_q8_buffer
        else:
            return SLOT_TO_BUFFER_DEFAULT.get(slot, "A_LAYER_INPUT")

    def record_write(self, buffer_name: str, op_id: int, dtype: str):
        """Record that an op wrote to a buffer."""
        if buffer_name in self.buffers:
            self.buffers[buffer_name].last_writer = op_id
            self.buffers[buffer_name].dtype = dtype


class MemoryPlanner:
    """
    Assigns physical buffers to ops based on dataflow.

    This replaces the ping-pong logic with explicit dataflow-based assignment.
    """

    def __init__(self):
        self.state = BufferState()
        self.assignments: Dict[int, Dict[str, Any]] = {}  # op_id -> assignments

    def plan(self, ir1_ops: List[Dict]) -> Dict[int, Dict[str, Any]]:
        """
        Plan buffer assignments for all ops.

        Args:
            ir1_ops: List of ops from IR1 with dataflow info

        Returns:
            Dict mapping op_id to buffer assignments:
            {
                op_id: {
                    "inputs": {input_name: {"buffer": "A_X", "dtype": "fp32"}},
                    "outputs": {output_name: {"buffer": "A_Y", "dtype": "fp32"}}
                }
            }
        """
        self.state = BufferState()
        self.assignments = {}

        current_layer = -1

        for op in ir1_ops:
            # Prefer idx (unique sequential index from IR Lower 1) over op_id
            # (which comes from IR1 and can collide after fusion/insertion)
            op_id = op.get("idx", op.get("op_id", -1))
            op_type = op.get("op", "")
            layer = op.get("layer", -1)
            dataflow = op.get("dataflow", {})

            # Track layer changes for debugging
            if layer != current_layer:
                current_layer = layer

            # Plan this op's buffer assignment
            assignment = self._plan_op(op_id, op_type, layer, dataflow)
            self.assignments[op_id] = assignment


        return self.assignments

    def _plan_op(self, op_id: int, op_type: str, layer: int, dataflow: Dict) -> Dict[str, Any]:
        """Plan buffer assignment for a single op."""

        inputs_assignment = {}
        outputs_assignment = {}

        # Get input assignments based on dataflow
        for input_name, input_info in dataflow.get("inputs", {}).items():
            buffer, dtype = self._get_input_buffer(op_type, input_name, input_info)
            inputs_assignment[input_name] = {"buffer": buffer, "dtype": dtype}

        # Get output assignments and update state
        for output_name, output_info in dataflow.get("outputs", {}).items():
            buffer, dtype = self._get_output_buffer(op_id, op_type, output_name, output_info)
            outputs_assignment[output_name] = {"buffer": buffer, "dtype": dtype}

        # For known op types without dataflow outputs (e.g., fused ops),
        # generate default output assignment so state tracking stays correct
        if not outputs_assignment and op_type in (
            "out_proj", "mlp_gate_up", "mlp_down", "residual_add",
            "rmsnorm", "dense_embedding_lookup", "residual_save",
        ):
            buffer, dtype = self._get_output_buffer(op_id, op_type, "y", {"dtype": "fp32"})
            outputs_assignment["y"] = {"buffer": buffer, "dtype": dtype}

        # Handle ping-pong for specific ops
        self._handle_pingpong(op_type)

        return {
            "inputs": inputs_assignment,
            "outputs": outputs_assignment,
            "op_type": op_type,
            "layer": layer,
        }

    def _get_input_buffer(self, op_type: str, input_name: str, input_info: Dict) -> Tuple[str, str]:
        """Determine which buffer an input should read from."""

        dtype = input_info.get("dtype", "fp32")

        # Special cases based on op type and input name
        if op_type == "residual_add":
            if input_name == "a":
                # 'a' is the current stream (from out_proj/bias_add or mlp_down/bias_add)
                # This should be the MAIN STREAM (FP32), not quantized
                return self.state.main_stream_buffer, "fp32"
            elif input_name == "b":
                # 'b' is the saved residual
                return "A_RESIDUAL", "fp32"

        elif op_type in ("q_proj", "k_proj", "v_proj"):
            # QKV projections read quantized input
            return self.state.main_stream_q8_buffer, dtype

        elif op_type == "out_proj":
            if dtype == "fp32":
                # Fused op: quantize absorbed, reads FP32 from attention scratch
                return "A_ATTN_SCRATCH", "fp32"
            # Unfused: reads quantized attention output
            return self.state.main_stream_q8_buffer, dtype

        elif op_type == "mlp_gate_up":
            if dtype == "fp32":
                # Fused op: quantize absorbed, reads FP32 from main stream
                return self.state.main_stream_buffer, "fp32"
            # Unfused: reads quantized input after second rmsnorm
            return self.state.main_stream_q8_buffer, dtype

        elif op_type == "mlp_down":
            # mlp_down reads quantized MLP intermediate
            return self.state.main_stream_q8_buffer, dtype

        elif op_type == "silu_mul":
            # silu_mul reads/writes MLP scratch (in-place)
            return "A_MLP_SCRATCH", "fp32"

        elif op_type in ("rmsnorm",):
            # rmsnorm reads main stream
            return self.state.main_stream_buffer, "fp32"

        elif op_type in ("quantize_input_0", "quantize_input_1"):
            # quantize reads from main stream (FP32)
            return self.state.main_stream_buffer, "fp32"

        elif op_type == "quantize_out_proj_input":
            # quantize_out_proj reads from attention scratch
            return "A_ATTN_SCRATCH", "fp32"

        elif op_type == "quantize_mlp_down_input":
            # quantize_mlp_down reads from MLP scratch
            return "A_MLP_SCRATCH", "fp32"

        elif op_type == "quantize_final_output":
            # quantize_final_output reads from main stream (footer rmsnorm output)
            return self.state.main_stream_buffer, "fp32"

        elif op_type == "attn":
            if input_name == "q":
                return "A_ATTN_SCRATCH", "fp32"
            else:
                # k, v come from KV cache
                return "kv_cache", "fp32"

        elif op_type == "rope_qk":
            # RoPE reads Q and K from attention scratch
            return "A_ATTN_SCRATCH", "fp32"

        elif op_type == "bias_add":
            # bias_add is in-place on current output
            return self.state.main_stream_buffer, "fp32"

        elif op_type == "logits":
            # logits reads quantized final hidden state
            return self.state.main_stream_q8_buffer, dtype

        # Default: use main stream
        return self.state.main_stream_buffer, dtype

    def _get_output_buffer(self, op_id: int, op_type: str, output_name: str,
                           output_info: Dict) -> Tuple[str, str]:
        """Determine which buffer an output should write to."""

        dtype = output_info.get("dtype", "fp32")

        # Special cases based on op type
        if op_type == "dense_embedding_lookup":
            # Embedding writes to main stream
            buffer = self.state.main_stream_buffer
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "rmsnorm":
            # RMSNorm writes to main stream
            buffer = self.state.main_stream_buffer
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type in ("quantize_input_0", "quantize_input_1"):
            # Quantize writes Q8 to the "other" buffer (ping-pong)
            buffer = self.state.main_stream_q8_buffer
            self.state.record_write(buffer, op_id, dtype)
            return buffer, dtype

        elif op_type == "quantize_out_proj_input":
            # Quantize attention output, write to main_stream_q8
            buffer = self.state.main_stream_q8_buffer
            self.state.record_write(buffer, op_id, dtype)
            return buffer, dtype

        elif op_type == "quantize_mlp_down_input":
            # Quantize MLP intermediate, write to main_stream_q8
            buffer = self.state.main_stream_q8_buffer
            self.state.record_write(buffer, op_id, dtype)
            return buffer, dtype

        elif op_type == "quantize_final_output":
            # Quantize footer rmsnorm output for logits, write to main_stream_q8
            buffer = self.state.main_stream_q8_buffer
            self.state.record_write(buffer, op_id, dtype)
            return buffer, dtype

        elif op_type in ("q_proj", "k_proj", "v_proj"):
            # QKV projections write to attention scratch
            buffer = "A_ATTN_SCRATCH"
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "rope_qk":
            # RoPE writes in-place to attention scratch
            return "A_ATTN_SCRATCH", "fp32"

        elif op_type == "attn":
            # Attention writes to attention scratch
            buffer = "A_ATTN_SCRATCH"
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "out_proj":
            # out_proj writes FP32 to main stream
            buffer = self.state.main_stream_buffer
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "mlp_gate_up":
            # mlp_gate_up writes to MLP scratch
            buffer = "A_MLP_SCRATCH"
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "silu_mul":
            # silu_mul writes in-place to MLP scratch
            return "A_MLP_SCRATCH", "fp32"

        elif op_type == "mlp_down":
            # mlp_down writes FP32 to main stream
            buffer = self.state.main_stream_buffer
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "residual_add":
            # residual_add writes to main stream
            buffer = self.state.main_stream_buffer
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        elif op_type == "bias_add":
            # bias_add is in-place
            buffer = self.state.main_stream_buffer
            return buffer, "fp32"

        elif op_type == "logits":
            # logits writes to logits buffer (must match codegen API)
            return "A_LOGITS", "fp32"

        elif op_type == "residual_save":
            # residual_save writes to residual buffer
            buffer = "A_RESIDUAL"
            self.state.record_write(buffer, op_id, "fp32")
            return buffer, "fp32"

        # Default: use main stream
        buffer = self.state.main_stream_buffer
        self.state.record_write(buffer, op_id, dtype)
        return buffer, dtype

    def _handle_pingpong(self, op_type: str):
        """Handle ping-pong buffer swapping after certain ops."""

        # Swap after quantize ops that feed into projections
        # This ensures the FP32 output buffer is ready for the next op
        if op_type in ("quantize_input_0", "quantize_input_1",
                       "quantize_out_proj_input", "quantize_mlp_down_input"):
            # After quantize, the Q8 is in main_stream_q8_buffer
            # The projection will read from there and write FP32 to main_stream_buffer
            pass  # No swap needed - projections read from q8 buffer, write to main buffer

        # After residual_add, we're ready for the next rmsnorm
        # Main stream now has FP32 result
        if op_type == "residual_add":
            # No swap needed - residual_add writes to main_stream_buffer
            pass


def plan_memory(ir1_ops: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """
    Plan memory buffer assignments for all ops.

    This is the main entry point for the memory planner.

    Args:
        ir1_ops: List of ops from IR1 with dataflow info

    Returns:
        Dict mapping op_id to buffer assignments
    """
    planner = MemoryPlanner()
    return planner.plan(ir1_ops)


def print_buffer_flow(ir1_ops: List[Dict], assignments: Dict[int, Dict[str, Any]],
                      layer: int = 0):
    """Print buffer flow for debugging."""

    print(f"\n{'='*70}")
    print(f"BUFFER FLOW - Layer {layer}")
    print(f"{'='*70}\n")

    layer_ops = [op for op in ir1_ops if op.get("layer") == layer]

    for op in layer_ops:
        op_id = op["op_id"]
        op_type = op["op"]
        assignment = assignments.get(op_id, {})

        inputs = assignment.get("inputs", {})
        outputs = assignment.get("outputs", {})

        print(f"Op {op_id}: {op_type}")

        if inputs:
            in_strs = [f"{k}={v['buffer']}({v['dtype']})" for k, v in inputs.items()]
            print(f"  IN:  {in_strs}")

        if outputs:
            out_strs = [f"{k}={v['buffer']}({v['dtype']})" for k, v in outputs.items()]
            print(f"  OUT: {out_strs}")

        print()


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memory_planner_v6_6.py <ir1_file.json> [layer]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        ir1_data = json.load(f)

    ir1_ops = ir1_data.get("ops", [])

    print(f"Planning memory for {len(ir1_ops)} ops...")
    assignments = plan_memory(ir1_ops)

    print(f"Generated {len(assignments)} buffer assignments")

    # Print buffer flow for specified layer or layer 0
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print_buffer_flow(ir1_ops, assignments, layer)
