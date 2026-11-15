"""Author declarations for existing checked operation interfaces.

These helpers construct circuit data only. Provider resolution, validation,
lowering, memory planning and code generation remain in the ordinary compiler.
They contain no model-family dispatch and execute no neural computation.
"""

def linear(name: str, source: str, output: str, input_width: int,
           output_width: int, rows: int, prefix: str) -> dict:
    return {
        "id": name, "op": "linear_rows_checked",
        "kernel": "linear_rows_checked_f32", "returns_status": True,
        "weight_refs": {"weight": f"{prefix}.weight", "bias": f"{prefix}.bias"},
        "graph_slots": {"inputs": {"input": source},
                        "outputs": {"output": output}},
        "params": {"M": rows, "K": input_width, "N": output_width,
                   "call_constants": {
                       "linear_input_elements": rows * input_width,
                       "linear_input_stride": input_width,
                       "linear_weight_elements": input_width * output_width,
                       "linear_weight_stride": input_width,
                       "linear_bias_elements": output_width,
                       "linear_output_elements": rows * output_width,
                       "linear_output_stride": output_width,
                       "linear_rows": rows,
                       "linear_input_channels": input_width,
                       "linear_output_channels": output_width}}}


def bind_runtime_scalars(op: dict, **bindings: str) -> None:
    op["consumes_runtime_lengths"] = sorted(set(bindings.values()))
    op["runtime_scalar_bindings"] = bindings
    constants = op["params"]["call_constants"]
    for argument in bindings:
        if argument not in constants:
            raise ValueError(f"{op['id']} has no scalar ABI argument {argument}")
        del constants[argument]


def runtime_extent(op_id: str, kind: str, source: str, target: str,
            output: str, *, offset: int = 0, factor: int = 1,
            capacity: int) -> dict:
    if kind == "runtime_extent_sum":
        return {
            "id": op_id, "op": kind, "kernel": "runtime_extent_sum_i32",
            "returns_status": True,
            "produces_runtime_lengths": {target: "valid_extent"},
            "graph_slots": {"inputs": {"values": "external:" + source},
                            "outputs": {"valid_extent": output}},
        }
    if kind not in ("runtime_extent_affine", "runtime_extent_scale"):
        raise ValueError("unsupported extent derivation")
    argument = ("extent_affine_source" if kind == "runtime_extent_affine"
                else "extent_scale_source")
    constants = ({"extent_affine_factor": factor,
                  "extent_affine_offset": offset,
                  "extent_affine_capacity": capacity} if kind == "runtime_extent_affine"
                 else {"extent_scale_factor": factor,
                       "extent_scale_capacity": capacity})
    return {
        "id": op_id, "op": kind, "kernel": kind + "_i32",
        "returns_status": True,
        "consumes_runtime_lengths": [source],
        "produces_runtime_lengths": {target: "valid_extent"},
        "runtime_scalar_bindings": {argument: source},
        "params": {"call_constants": constants},
        "graph_slots": {"inputs": {}, "outputs": {"valid_extent": output}},
    }



def declared_f32_weight_shapes(graph):
    """Shapes for the supported checked table, row and positional interfaces.

    This is an explicit bounded authoring adapter, not shape inference for all
    kernel maps. Unknown weight ports and inconsistent shared weights reject.
    """
    shapes={}
    for block in graph['block_types'].values():
        for op in block.get('body',{}).get('ops',[]):
            for port,name in op.get('weight_refs',{}).items():
                p=op['params']
                if port=='table':shape=[p['V'],p['C']]
                elif port in ('gamma','frequencies'):shape=[p['C']]
                elif port=='positions':shape=[p['R']]
                elif port=='bias':shape=[p['N']]
                elif port=='weight':shape=[p['N'],p['K']]
                else:raise ValueError('unsupported declared weight port: '+port)
                if any(type(n) is not int or n<1 for n in shape):raise ValueError('invalid declared weight shape')
                if name in shapes and shapes[name]!=shape:raise ValueError('conflicting shapes for shared weight '+name)
                shapes[name]=shape
    return shapes


def execution_port(graph,buffer,*,leading_slice=None):
    """Declare a named port; the compiler independently validates its shape/ABI."""
    shape=list(graph['activation_buffers'][buffer]['shape'])
    result={'buffer':buffer,'shape':shape}
    if leading_slice is not None:
        start,length=leading_slice
        result['slice']={'axis':0,'start':start,'length':length};shape[0]=length
    return result
