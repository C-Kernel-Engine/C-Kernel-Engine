"""Single-speaker embedding input for the pinned dense Ming TTS backbone."""
import numpy as np

from version.v8.scripts.checked_circuit_authoring_v8 import (
    bind_runtime_scalars, execution_port, linear,
)

SPEAKER_DIM = 192


def add_speaker_prompt(graph):
    """Project CAM++ features and replace the reserved token before prefill."""
    width = graph['contract']['runtime_invariants']['width']
    capacity = graph['contract']['runtime_invariants']['capacity']
    graph['name'] += '__speaker'
    graph['contract']['runtime_invariants']['speaker_input'] = 'camplus_192'
    ops = graph['block_types']['embedding']['body']['ops']
    lookup = ops[0]
    lookup['graph_slots']['outputs']['output'] = 'token_rows'
    graph['semantic_checkpoints']['exports'][lookup['id']]['checkpoints'][0]['tensor'] = 'token_rows'
    for name, shape in {'token_rows': [capacity, width], 'speaker': [1, SPEAKER_DIM],
                        'speaker_row': [1, width], 'token_indices': [capacity],
                        'speaker_indices': [capacity]}.items():
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    project = linear('project_speaker', 'external:speaker', 'speaker_row',
                     SPEAKER_DIM, width, 1, 'spk_head')
    splice = {
        'id': 'bind_speaker', 'op': 'indexed_sum_rows_checked',
        'kernel': 'indexed_sum_rows_checked_f32', 'returns_status': True,
        'graph_slots': {'inputs': {'left': 'token_rows', 'right': 'speaker_row',
                                  'left_indices': 'external:token_indices',
                                  'right_indices': 'external:speaker_indices'},
                        'outputs': {'output': 'external:embeddings'}},
        'params': {'L': capacity, 'K': 1, 'R': capacity, 'C': width,
                   'call_constants': {
                       'left_elements': capacity * width, 'left_stride': width, 'left_rows': capacity,
                       'right_elements': width, 'right_stride': width, 'right_rows': 1,
                       'left_index_elements': capacity, 'right_index_elements': capacity,
                       'output_elements': capacity * width, 'output_stride': width,
                       'rows': capacity, 'channels': width}}}
    bind_runtime_scalars(splice, rows='embed_rows', left_rows='embed_rows')
    ops.extend([project, splice])
    graph['native_entry']['segments'][0]['last_op'] = splice['id']
    for name in ('speaker', 'token_indices', 'speaker_indices'):
        graph['execution_exports']['ports'][name] = execution_port(graph, name)


def check_vector(speaker):
    if (not isinstance(speaker, np.ndarray) or speaker.dtype != np.float32
            or speaker.shape != (SPEAKER_DIM,) or not np.isfinite(speaker).all()):
        raise ValueError('expected 192 finite FP32 CAM++ speaker values')


def bind_speaker(ports, arena, token_ids, position, speaker):
    # These three IDs come from the SHA-pinned dense tokenizer's speaker delimiters.
    count = len(token_ids)
    if (type(position) is not int or not 0 < position < count - 1
            or tuple(token_ids[position-1:position+2]) != (151652, 151654, 151653)
            or token_ids.count(151654) != 1):
        raise ValueError('expected one delimited speaker slot in the prompt')
    check_vector(speaker)

    def view(name, dtype):
        port = ports[name]
        return np.ndarray(port['shape'], dtype, buffer=arena, offset=port['offset'])
    tokens = view('token_indices', np.int32)[:count]
    indices = view('speaker_indices', np.int32)[:count]
    tokens[:] = np.arange(count)
    indices.fill(-1)
    tokens[position] = -1
    indices[position] = 0
    view('speaker', np.float32)[0] = speaker
