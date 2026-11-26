"""Bind an upstream Base speaker x-vector to the checked talker graph."""
import numpy as np

WIDTH = 1024
SPEAKER_ROW = 4


def check_speaker(vector):
    if (not isinstance(vector, np.ndarray) or vector.dtype != np.float32
            or vector.shape != (WIDTH,) or not np.isfinite(vector).all()):
        raise ValueError('expected a finite FP32 speaker vector of shape (1024,)')
    return vector.copy()


def add_base_speaker(graph):
    """Replace codec row 4 before the existing text/codec prompt assembly."""
    ops = graph['block_types']['text_projection']['body']['ops']
    index = next(i for i, op in enumerate(ops) if op['id'] == 'prompt_assembly')
    assembly = ops[index]['graph_slots']['inputs']
    if assembly['right'] != 'codec_embedding_token':
        raise ValueError('unexpected talker prompt layout')
    for name, shape in {'speaker_vector': [1, WIDTH], 'voice_codec_rows': [7, WIDTH],
                        'voice_left_indices': [7], 'voice_right_indices': [7]}.items():
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    ops.insert(index, {
        'id': 'bind_base_speaker', 'op': 'indexed_sum_rows_checked',
        'kernel': 'indexed_sum_rows_checked_f32', 'returns_status': True,
        'graph_slots': {'inputs': {'left': assembly['right'], 'right': 'external:speaker_vector',
                                  'left_indices': 'external:voice_left_indices',
                                  'right_indices': 'external:voice_right_indices'},
                        'outputs': {'output': 'voice_codec_rows'}},
        'params': {'L': 7, 'K': 1, 'R': 7, 'C': WIDTH, 'call_constants': {
            'left_elements': 7 * WIDTH, 'left_stride': WIDTH, 'left_rows': 7,
            'right_elements': WIDTH, 'right_stride': WIDTH, 'right_rows': 1,
            'left_index_elements': 7, 'right_index_elements': 7,
            'output_elements': 7 * WIDTH, 'output_stride': WIDTH,
            'rows': 7, 'channels': WIDTH}}})
    assembly['right'] = 'voice_codec_rows'
    graph['name'] += '__base_xvector'
    graph['contract']['runtime_invariants'].update(
        speaker_input='base_xvector_fp32', reference_scope='experimental Base English x-vector only')


def bind_base_speaker(arena, acts, vector):
    def view(name, dtype, count):
        return np.ndarray(count, dtype, buffer=arena, offset=acts[name]['abs_offset'])
    left = view('voice_left_indices', np.int32, 7)
    right = view('voice_right_indices', np.int32, 7)
    left[:] = np.arange(7)
    left[SPEAKER_ROW] = -1
    right.fill(-1)
    right[SPEAKER_ROW] = 0
    view('speaker_vector', np.float32, WIDTH)[:] = vector
