"""Isolated HTTP worker contract for one Nemotron Nano 12B v2 VL tile."""
from array import array
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from server.vision_bundle import sha


PLACEHOLDER = '<image>'
IMAGE_IDS = (131073, 131072, 131074)
STOP_ID = 12
PROJECTOR_SHAPES = {'mlp1.0.weight': (5120,), 'mlp1.1.weight': (20480, 5120),
                    'mlp1.3.weight': (5120, 20480)}


class VisionWeights:
    """Read only the pinned vision tensors, without importing a model framework."""
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as source:
            length = int.from_bytes(source.read(8), 'little')
            if not 2 <= length <= 1 << 20:
                raise ValueError('invalid vision safetensors header length')
            self.entries = json.loads(source.read(length))
        if not isinstance(self.entries, dict):
            raise ValueError('invalid vision tensor index')
        self.base = 8 + length
        self.limit = self.path.stat().st_size - self.base

    def read(self, name, *, dtype='BF16'):
        import numpy as np

        entry = self.entries[name]
        shape = entry['shape']
        start, end = entry['data_offsets']
        storage = np.dtype({'BF16': '<u2', 'F32': '<f4'}[dtype])
        if (entry['dtype'] != dtype or not 1 <= len(shape) <= 4
                or any(type(n) is not int or n <= 0 for n in shape)
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= self.limit
                or end - start != math.prod(shape) * storage.itemsize):
            raise ValueError('invalid pinned vision tensor descriptor')
        return np.memmap(self.path, dtype=storage, mode='r', offset=self.base + start, shape=tuple(shape))


def prompt_segments(tokenizer, rendered, rows):
    if not rendered or rendered.count(PLACEHOLDER) != 1:
        raise ValueError('expected exactly one publisher image placeholder')
    begin, image, end = IMAGE_IDS
    ids = tokenizer.encode(rendered.replace(PLACEHOLDER, '<img>' + PLACEHOLDER * rows + '</img>'),
                           add_bos=False)
    left = ids.index(begin)
    right = left + rows + 1
    if (left == 0 or right + 1 >= len(ids) or ids[0] != 1
            or ids[left:right + 1] != [begin] + [image] * rows + [end]
            or any(token in IMAGE_IDS for token in ids[:left] + ids[right + 1:])):
        raise ValueError('invalid Nemotron linear visual span')
    return ids[:left + 1], ids[right:]


def run_nano_vl(profile, payload, emit):
    for key in ('CK_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(payload['threads'])
    import numpy as np
    from PIL import Image
    from server.live import _render_with_chat_templates
    from version.v8.scripts.nemotron_radio_checked_v8 import CompiledRadioEncoder, RESOLUTION, widen
    from version.v8.scripts.nemotron_vl_checked_v8 import (
        CompiledProjector, DECODER_WIDTH, VISUAL_ROWS, validate_publisher_config,
    )
    # Match the imports used by the existing worker and the generated bridge CLI.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'version/v8/scripts'))
    from run_multimodal_bridge_v8 import (
        GGUFTokenizer, _load_decoder_lib, _load_prebuilt_decoder_runtime,
        _resolved_symbol_library, _run_decoder,
    )

    started = time.monotonic()
    encdir, decdir = Path(profile['encoder']), Path(profile['decoder'])
    config = json.loads((encdir / 'config.json').read_text())
    validate_publisher_config(config)
    runtime = _load_prebuilt_decoder_runtime(decdir)
    cfg = json.loads((decdir / 'layout_decode.json').read_text())['config']
    if (cfg.get('arch') != 'nemotron_h' or cfg.get('num_layers') != 62
            or runtime['embed_dim'] != DECODER_WIDTH or runtime['input_embed_dim'] != DECODER_WIDTH
            or runtime['vocab_size'] != 132096):
        raise ValueError('unsupported Nano VL decoder layout')
    capacity = runtime['context_length']
    template = (decdir / 'chat_template.jinja').read_text()
    tokenizer = GGUFTokenizer.from_gguf(profile['tokenizer_gguf'])
    messages = payload.get('messages', [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': 'Describe.'},
    ]}])
    rendered = _render_with_chat_templates(template, None, messages, None,
                                           renderer_tokens={'bos_token': '<s>', 'eos_token': '<SPECIAL_12>'})
    before, after = prompt_segments(tokenizer, rendered, VISUAL_ROWS)
    budget = payload.get('max_tokens', 1)
    prompt_tokens = len(before) + VISUAL_ROWS + len(after)
    if prompt_tokens + budget > capacity:
        emit({'type': 'error', 'code': 'context_length_exceeded',
              'message': 'Rendered text, visual rows and output exceed capacity'})
        return

    blas_path = (encdir / 'libblas.so').resolve()
    blas = ctypes.CDLL(str(blas_path))
    if (_resolved_symbol_library(blas, 'cblas_gemm_bf16bf16f32') != blas_path
            or sha(blas_path) != profile['files']['encoder/libblas.so']):
        raise ValueError('loaded projector BLAS does not match pinned artifact')
    os.environ['CK_BLAS_LIBRARY'] = str(blas_path)
    lib = _load_decoder_lib(runtime['so_path'], engine_so=runtime['engine_so'])
    lib.ck_model_get_context_window.restype = ctypes.c_int
    if lib.ck_model_get_context_window() != capacity or lib.ck_model_get_vocab_size() != runtime['vocab_size']:
        raise ValueError('loaded decoder geometry does not match pinned layout')
    identities = {'projector_blas': profile['files']['encoder/libblas.so']}
    for symbol, name in [('ck_model_forward_segments_grid_ex', 'libmodel.so'),
                         ('ck_set_num_threads', 'libckernel_engine.so'),
                         ('ck_true_bpe_decode', 'libckernel_tokenizer.so')]:
        identity = sha(_resolved_symbol_library(lib, symbol))
        if identity != profile['files']['decoder/' + name]:
            raise ValueError('loaded decoder does not match pinned artifact')
        identities[symbol] = identity

    weights = VisionWeights(encdir / 'vision.safetensors')
    norm = 'vision_model.radio_model.input_conditioner.norm_'
    mean, std = (weights.read(norm + name, dtype='F32').reshape(3) for name in ('mean', 'std'))
    projection = {name: weights.read(name) for name in PROJECTOR_SHAPES}
    if any(projection[name].shape != shape for name, shape in PROJECTOR_SHAPES.items()):
        raise ValueError('invalid Nano VL projector geometry')
    with tempfile.TemporaryDirectory(prefix='radio-build-', dir=payload['work_dir']) as work:
        root = Path(work)
        encoder = CompiledRadioEncoder(root / 'radio', config, weights.read, mean, std)
        projector = CompiledProjector(root / 'projection', config, projection, cblas=True)
        for role, relative in [('radio_patch', 'radio/patch'), ('radio_position', 'radio/position'),
                               ('radio_block', 'radio/block'), ('projector_pack', 'projection/pack'),
                               ('projector_weighted', 'projection/weighted')]:
            identities[role] = sha(root / relative / 'generated.so')
        if payload.get('probe'):
            emit({'type': 'complete', 'capabilities': {'experimental': True, 'vision': True,
                'api': ['chat_completions'], 'tools': False, 'max_images': 1, 'context_length': capacity,
                'visual_rows': VISUAL_ROWS, 'visual_width': DECODER_WIDTH, 'resident': False,
                'image_geometry': [RESOLUTION, RESOLUTION]}, 'bundle_identity': profile['identity'],
                'loaded_libraries': identities})
            return

        emit({'type': 'stage', 'stage': 'encoder'})
        encoder_start = time.monotonic()
        image_path = Path(payload['image_path'])
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert('RGB').resize((RESOLUTION, RESOLUTION), Image.Resampling.BICUBIC))
        prefix = array('f')
        prefix.frombytes(widen(projector(encoder(rgb))).tobytes())
        encoder_seconds = time.monotonic() - encoder_start
    del encoder, projector

    emit({'type': 'stage', 'stage': 'decoder'})
    result = _run_decoder(runtime, prefix, VISUAL_ROWS, after, tokens_before=before,
                          prefix_embed_dim=DECODER_WIDTH, prefix_text_pos=len(before) + VISUAL_ROWS,
                          bridge_runtime='decode-staged', bridge_generation_mode='incremental-decode',
                          tokenizer=tokenizer, stop_token_ids=[STOP_ID], max_tokens=budget)
    count = len(result['generated_token_ids'])
    stop = result['generation_stop_reason']
    if stop not in ('stop_token', 'max_tokens'):
        raise ValueError('unexpected decoder finish reason')
    if result['generated_text']:
        emit({'type': 'delta', 'text': result['generated_text']})
    emit({'type': 'complete', 'finish_reason': 'stop' if stop == 'stop_token' else 'length',
          'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': count,
                    'total_tokens': prompt_tokens + count},
          'evidence': {'bundle_identity': profile['identity'], 'loaded_libraries': identities,
                       'image_sha256': sha(image_path), 'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                       'encoder_seconds': encoder_seconds,
                       'prefill_ms': result['timings']['decoder_forward_mixed_ms'],
                       'decode_ms': result['timings']['decoder_generation_ms'],
                       'total_seconds': time.monotonic() - started}})
