"""Declare Bison's first packed token-to-Q/K/V circuit fragment."""
from version.v8.scripts.checked_bf16_qkv_authoring_v8 import bf16_qkv_ingress


def build_circuit(rows, *, table_rows, width, q_width, kv_width, epsilon):
    """Bind a compact embedding-table input prepared from checkpoint rows.

    Embedding rows already include the checkpoint's BF16 embedding multiplier.
    Indices address that table, in the order of the original packed token IDs.
    """
    if type(table_rows) is not int or table_rows < 1:
        raise ValueError("embedding table must have at least one row")
    graph, config = bf16_qkv_ingress(
        name="bison_backbone_ingress", entry="ck_bison_backbone_ingress",
        rows=rows, width=width, q_width=q_width, kv_width=kv_width, epsilon=epsilon,
        norm="rmsnorm_rows_weighted_single_round_bf16_checked",
    )
    buffers = graph["activation_buffers"]
    buffers.update(token_rows={"shape": [rows]}, embedding_table={"shape": [table_rows, width]})
    graph["activation_bindings"].update(token_rows="token_rows", embedding_table="embedding_table")
    config["activation_buffer_dtypes"].update(token_rows="i32", embedding_table="bf16")
    ops = graph["block_types"]["block"]["body"]["ops"]
    ops[0]["graph_slots"]["inputs"]["input"] = "input"
    gather = "gather_rows_bf16_checked"
    ops.insert(0, {
        "id": "token_embedding", "op": gather, "kernel": gather, "returns_status": True,
        "graph_slots": {
            "inputs": {"input": "external:embedding_table", "indices": "external:token_rows"},
            "outputs": {"output": "input"},
        },
        "params": {"R": table_rows, "S": rows, "D": width, "call_constants": {
            "input_elements": table_rows * width, "index_elements": rows,
            "output_elements": rows * width, "input_rows": table_rows,
            "output_rows": rows, "width": width,
        }},
    })
    return graph, config
