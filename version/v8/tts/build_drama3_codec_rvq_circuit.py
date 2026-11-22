"""Declare Drama 3's semantic and residual codebook reconstruction."""
from version.v8.scripts.checked_circuit_authoring_v8 import linear


def build_circuit(frames: int, *, vocabularies=(4096,) + (1024,) * 9,
                  book_width: int = 8, channels: int = 1024) -> tuple[dict, dict]:
    """Build a fixed-frame prefix ending at the input to the codec transformer."""
    if type(frames) is not int or frames < 1:
        raise ValueError("frames must be a positive integer")
    if len(vocabularies) != 10 or any(
        type(size) is not int or size < 1 for size in (*vocabularies, book_width, channels)
    ):
        raise ValueError("expected one semantic and nine residual codebooks with positive dimensions")

    shapes = {}
    dtypes = {}
    operations = []
    for index, vocabulary in enumerate(vocabularies):
        ids, embedding, projected = (f"ids{index}", f"embedding{index}", f"projection{index}")
        weight = f"book{index}"
        shapes[ids] = [frames]
        dtypes[ids] = "i32"
        shapes[embedding] = [frames, book_width]
        shapes[projected] = [frames, channels]
        operations.extend([
            {
                "id": embedding, "op": "embedding_lookup_token_major_checked",
                "kernel": "audio_text_embedding_token_major_f32", "returns_status": True,
                "weight_refs": {"table": f"{weight}.table"},
                "graph_slots": {"inputs": {"ids": f"external:{ids}"}, "outputs": {"output": embedding}},
                "params": {"T": frames, "C": book_width, "V": vocabulary,
                           "call_constants": {"tokens": frames, "vocabulary": vocabulary,
                               "channels": book_width, "id_elements": frames,
                               "table_elements": vocabulary * book_width, "table_stride": book_width,
                               "output_elements": frames * book_width, "output_stride": book_width}},
            },
            linear(projected, embedding, projected, book_width, channels, frames, weight),
        ])

    def add(left, right, output):
        shapes[output] = [frames, channels]
        elements = frames * channels
        operations.append({
            "id": output, "op": "add_rows_checked", "kernel": "add_rows_checked_f32",
            "returns_status": True,
            "graph_slots": {"inputs": {"left": left, "right": right}, "outputs": {"output": output}},
            "params": {"N": elements, "call_constants": {
                "elements": elements, "left_elements": elements, "right_elements": elements,
                "output_elements": elements}},
        })

    residual = "projection1"
    for index in range(2, 10):
        output = f"residual{index}"
        add(residual, f"projection{index}", output)
        residual = output
    add("projection0", residual, "quantized")

    graph = {
        "version": 3, "name": "drama3_codec_rvq", "family": "bounded_graph",
        "checked_native_entry": True,
        "activation_buffers": {name: {"shape": shape} for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "contract": {"runtime_invariants": {"inference_only": True}},
        "native_entry": {
            "function": "ck_drama3_codec_rvq",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["rvq"],
        "block_types": {"rvq": {"sequence": ["header", "body", "footer"],
                                "header": [], "footer": [],
                                "body": {"type": "dense", "ops": operations}}},
    }
    config = {"num_layers": 1, "embed_dim": channels, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": channels, "intermediate_size": channels, "context_length": frames,
              "max_seq_len": frames, "activation_buffer_dtypes": dtypes}
    return graph, config
