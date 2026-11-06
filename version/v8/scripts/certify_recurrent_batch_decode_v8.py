#!/usr/bin/env python3
"""Certify one opt-in recurrent two-row decode bundle against isolated decode.

This is an asset-backed local gate. It checks complete logical KV, convolution,
and SSM regions after every step, along with full logits and positions.
"""
import argparse
import ctypes as c
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

NAMES = ('kv_cache', 'recurrent_conv_state', 'recurrent_ssm_state')

class Row(c.Structure):
    _fields_ = [('sequence_handle', c.c_uint64), ('token', c.c_int32),
                ('position', c.c_int32), ('row_offset', c.c_uint32),
                ('token_count', c.c_uint32), ('logits', c.POINTER(c.c_float))]

def aligned(size, alignment):
    storage = c.create_string_buffer(size + alignment - 1)
    ptr = (c.addressof(storage) + alignment - 1) & ~(alignment - 1)
    return storage, c.c_void_p(ptr)

def check(status, label):
    if status != 0:
        raise RuntimeError(f'{label}: {status}')

def bind(model):
    signatures = {
        'ck_model_init': ([c.c_char_p], c.c_int),
        'ck_model_free': ([], None),
        'ck_model_get_vocab_size': ([], c.c_int),
        'ck_model_get_capabilities': ([], c.c_uint64),
        'ck_model_decode': ([c.c_int32, c.POINTER(c.c_float)], c.c_int),
        'ck_model_get_active_tokens': ([], c.c_int),
        'ck_model_kv_cache_reset': ([], None),
        'ck_model_sequence_state_default': ([], c.c_uint64),
        'ck_model_sequence_state_requirements': ([c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)], c.c_int),
        'ck_model_sequence_state_create': ([c.c_void_p, c.c_size_t, c.POINTER(c.c_uint64)], c.c_int),
        'ck_model_sequence_state_activate': ([c.c_uint64], c.c_int),
        'ck_model_sequence_state_destroy': ([c.c_uint64], c.c_int),
        'ck_model_get_named_activation_ptr': ([c.c_char_p], c.c_size_t),
        'ck_model_batch_decode_workspace': ([c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)], c.c_int),
        'ck_model_batch_decode_projection_groups': ([], c.c_int),
        'ck_model_batch_decode_shared_layers': ([], c.c_int),
        'ck_model_batch_decode_snapshot_copy_bytes': ([], c.c_size_t),
        'ck_model_batch_decode_packing_copy_bytes': ([], c.c_size_t),
        'ck_model_batch_decode_last_copy_ns': ([], c.c_uint64),
        'ck_model_set_cancel_flag': ([c.POINTER(c.c_int)], None),
        'ck_model_decode_batch2': ([c.POINTER(Row), c.c_size_t, c.c_void_p, c.c_size_t], c.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(model, name)
        fn.argtypes, fn.restype = args, result

def certify(bundle: Path, *, swap_row_order: bool = False,
            expected_projection_groups: int = 1,
            expected_shared_layers: int = 1) -> dict:
    layout = json.loads((bundle / 'layout_decode.json').read_text())
    sizes = {row['name']: int(row['size']) for row in layout['memory']['activations']['buffers'] if row['name'] in NAMES}
    assert set(sizes) == set(NAMES), sizes
    model = c.CDLL(str(bundle / 'libmodel.so'), mode=c.RTLD_GLOBAL)
    bind(model)
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if line.split()[-1].startswith('/')}
    libraries = {}
    for name in ('libmodel.so', 'libckernel_engine.so', 'libckernel_tokenizer.so'):
        path = (bundle / name).resolve()
        if path not in mapped:
            raise AssertionError(f'loaded library backing differs: {name}')
        libraries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    model.ck_model_generated_source_sha256.restype = c.c_char_p
    source = (bundle / 'model_v8.c').read_bytes()
    marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
    prefix, found, tail = source.rpartition(marker)
    claimed = model.ck_model_generated_source_sha256()
    if not found or not claimed or tail != claimed + b'";\n}\n' or hashlib.sha256(prefix).hexdigest().encode() != claimed:
        raise AssertionError('loaded generated-source identity mismatch')
    check(model.ck_model_init(str(bundle / 'weights.bump').encode()), 'init')
    try:
        actual_groups = model.ck_model_batch_decode_projection_groups()
        if actual_groups != expected_projection_groups:
            raise AssertionError(
                f'expected {expected_projection_groups} shared projection groups, '
                f'loaded entry has {actual_groups}')
        actual_layers = model.ck_model_batch_decode_shared_layers()
        if actual_layers != expected_shared_layers:
            raise AssertionError(
                f'expected {expected_shared_layers} shared layers, '
                f'loaded entry has {actual_layers}')
        assert model.ck_model_get_capabilities() & (1 << 17)
        vocab = model.ck_model_get_vocab_size()
        if vocab <= 300:
            raise ValueError('bundle vocabulary cannot run the pinned token fixture')
        def state():
            result = {}
            for name in NAMES:
                address = model.ck_model_get_named_activation_ptr(name.encode())
                assert address, name
                region = (c.c_ubyte * sizes[name]).from_address(address)
                result[name] = hashlib.sha256(region).hexdigest()
            return result
        tokens = {'a': (100, 101, 102), 'b': (200, 201, 202, 203, 204),
                  'c': (300,)}
        default = model.ck_model_sequence_state_default()
        baseline = {}
        for name, sequence in tokens.items():
            check(model.ck_model_sequence_state_activate(default), 'default activate')
            model.ck_model_kv_cache_reset()
            baseline[name] = []
            for token in sequence:
                output = (c.c_float * vocab)()
                start = time.perf_counter_ns()
                check(model.ck_model_decode(token, output), 'isolated decode')
                native_ms = (time.perf_counter_ns() - start) / 1_000_000
                if not all(math.isfinite(value) for value in output):
                    raise AssertionError(f'{name} isolated logits are nonfinite')
                baseline[name].append((bytes(output), model.ck_model_get_active_tokens(),
                                       state(), native_ms))
        state_bytes, state_align = c.c_size_t(), c.c_size_t()
        check(model.ck_model_sequence_state_requirements(c.byref(state_bytes), c.byref(state_align)), 'state requirements')
        arenas, addresses, handles = {}, {}, {}
        for name in ('a', 'b'):
            storage, pointer = aligned(state_bytes.value, state_align.value)
            handle = c.c_uint64()
            check(model.ck_model_sequence_state_create(pointer, state_bytes, c.byref(handle)), 'create')
            arenas[name] = storage
            addresses[name] = pointer
            handles[name] = handle.value
        work_bytes, work_align = c.c_size_t(), c.c_size_t()
        check(model.ck_model_batch_decode_workspace(c.byref(work_bytes), c.byref(work_align)), 'workspace requirements')
        work_storage, work = aligned(work_bytes.value, work_align.value)
        def compare(name, step, output):
            check(model.ck_model_sequence_state_activate(handles[name]), 'activate')
            expected_logits, expected_pos, expected_state, _ = baseline[name][step]
            if not all(math.isfinite(value) for value in output):
                raise AssertionError(f'{name} step {step} batch logits are nonfinite')
            if bytes(output) != expected_logits:
                a, b = (c.c_float * vocab).from_buffer_copy(expected_logits), output
                worst = max(range(vocab), key=lambda i: abs(a[i]-b[i]))
                raise AssertionError(f'{name} step {step} logits differ: max={abs(a[worst]-b[worst])} index={worst}')
            if model.ck_model_get_active_tokens() != expected_pos:
                raise AssertionError(f'{name} step {step} position mismatch')
            actual_state = state()
            if actual_state != expected_state:
                raise AssertionError(f'{name} step {step} state mismatch: {[k for k in NAMES if actual_state[k] != expected_state[k]]}')
        # B has one earlier token; both subsequent rows advance in one shared call.
        check(model.ck_model_sequence_state_activate(handles['b']), 'B activate')
        b0 = (c.c_float * vocab)()
        check(model.ck_model_decode(tokens['b'][0], b0), 'B prefix')
        compare('b', 0, b0)
        steps = []
        for index in range(3):
            a_out, b_out = (c.c_float * vocab)(), (c.c_float * vocab)()
            order = ('b', 'a') if swap_row_order else ('a', 'b')
            descriptions = {
                'a': (tokens['a'][index], index, a_out),
                'b': (tokens['b'][index+1], index+1, b_out),
            }
            rows = (Row * 2)(*(Row(handles[name], descriptions[name][0],
                                   descriptions[name][1], offset, 1,
                                   descriptions[name][2])
                                 for offset, name in enumerate(order)))
            start = time.perf_counter_ns()
            check(model.ck_model_decode_batch2(rows, 2, work, work_bytes), f'batch step {index}')
            batch_ms = (time.perf_counter_ns() - start) / 1_000_000
            copy_ms = model.ck_model_batch_decode_last_copy_ns() / 1_000_000
            compare('a', index, a_out)
            compare('b', index+1, b_out)
            steps.append({'step': index, 'a_position': index+1, 'b_position': index+2,
                          'full_logits_exact': True, 'kv_conv_ssm_exact': True,
                          'isolated_pair_native_ms': baseline['a'][index][3] + baseline['b'][index+1][3],
                          'batch_native_ms': batch_ms, 'profiled_copy_ms': copy_ms})
        # A cancelled step must leave both state regions and positions intact.
        check(model.ck_model_sequence_state_activate(handles['a']), 'A activate before cancel')
        before_cancel_a = state()
        before_pos_a = model.ck_model_get_active_tokens()
        check(model.ck_model_sequence_state_activate(handles['b']), 'B activate before cancel')
        before_cancel_b = state()
        before_pos_b = model.ck_model_get_active_tokens()
        cancel = c.c_int(1)
        model.ck_model_set_cancel_flag(c.byref(cancel))
        a_cancel, b_cancel = (c.c_float * vocab)(), (c.c_float * vocab)()
        cancelled_rows = (Row * 2)(
            Row(handles['a'], 103, 3, 0, 1, a_cancel),
            Row(handles['b'], 204, 4, 1, 1, b_cancel))
        cancelled_status = model.ck_model_decode_batch2(
            cancelled_rows, 2, work, work_bytes)
        cancel.value = 0
        model.ck_model_set_cancel_flag(None)
        check(model.ck_model_sequence_state_activate(handles['a']), 'A activate after cancel')
        unchanged_a = state() == before_cancel_a and model.ck_model_get_active_tokens() == before_pos_a
        check(model.ck_model_sequence_state_activate(handles['b']), 'B activate after cancel')
        unchanged_b = state() == before_cancel_b and model.ck_model_get_active_tokens() == before_pos_b
        if cancelled_status != -2 or not unchanged_a or not unchanged_b:
            raise AssertionError('cancelled batch mutated a participating state')

        retired = handles['a']
        check(model.ck_model_sequence_state_destroy(retired), 'retire A')
        stale_status = model.ck_model_decode_batch2(
            cancelled_rows, 2, work, work_bytes)
        check(model.ck_model_sequence_state_activate(handles['b']), 'B activate after stale call')
        if (stale_status != -2 or state() != before_cancel_b or
                model.ck_model_get_active_tokens() != before_pos_b):
            raise AssertionError('stale handle mutated surviving state')
        fresh = c.c_uint64()
        check(model.ck_model_sequence_state_create(
            addresses['a'], state_bytes, c.byref(fresh)), 'reuse A slot for C')
        handles['c'] = fresh.value
        if handles['c'] == retired:
            raise AssertionError('slot reuse retained stale handle')
        c_out, b_out = (c.c_float * vocab)(), (c.c_float * vocab)()
        reuse_order = ('b', 'c') if swap_row_order else ('c', 'b')
        descriptions = {'c': (300, 0, c_out), 'b': (204, 4, b_out)}
        reuse_rows = (Row * 2)(*(Row(handles[name], descriptions[name][0],
                                   descriptions[name][1], offset, 1,
                                   descriptions[name][2])
                                 for offset, name in enumerate(reuse_order)))
        check(model.ck_model_decode_batch2(reuse_rows, 2, work, work_bytes),
              'reuse step')
        compare('c', 0, c_out)
        compare('b', 4, b_out)
        for handle in (handles['b'], handles['c']):
            check(model.ck_model_sequence_state_destroy(handle), 'destroy')
        engine = c.CDLL(str(bundle / 'libckernel_engine.so'))
        engine.ck_threadpool_global.argtypes = []
        engine.ck_threadpool_global.restype = c.c_void_p
        engine.ck_threadpool_n_threads.argtypes = [c.c_void_p]
        engine.ck_threadpool_n_threads.restype = c.c_int
        resolved_workers = engine.ck_threadpool_n_threads(engine.ck_threadpool_global())
        if resolved_workers < 1:
            raise AssertionError('invalid loaded thread-pool width')
        weights_path = bundle / 'weights.bump'
        with weights_path.open('rb') as weights_file:
            weights_sha256 = hashlib.file_digest(weights_file, 'sha256').hexdigest()
        return {'status': 'PASS', 'source_sha256': hashlib.sha256(source).hexdigest(),
                'loaded_libraries': libraries,
                'weights_path': str(weights_path.resolve()),
                'weights_bytes': weights_path.stat().st_size,
                'weights_sha256': weights_sha256,
                'weights_manifest_sha256': hashlib.sha256(
                    (bundle / 'weights_manifest.json').read_bytes()).hexdigest(),
                'layout_sha256': hashlib.sha256((bundle / 'layout_decode.json').read_bytes()).hexdigest(),
                'tokens': tokens, 'row_order': list(order),
                'projection_groups': actual_groups,
                'shared_layers': actual_layers,
                'requested_threads': os.getenv('CK_NUM_THREADS'),
                'resolved_threads': resolved_workers,
                'cpu': next((line.partition(':')[2].strip() for line in
                             Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')), platform.processor()),
                'state_bytes_per_sequence': state_bytes.value,
                'workspace_bytes': work_bytes.value,
                'planned_snapshot_copy_bytes': model.ck_model_batch_decode_snapshot_copy_bytes(),
                'planned_packing_copy_bytes': model.ck_model_batch_decode_packing_copy_bytes(),
                'cancellation_unchanged': True, 'stale_handle_rejected': True,
                'slot_reuse_exact': True,
                'steps': steps}
    finally:
        model.ck_model_free()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--swap-row-order', action='store_true')
    parser.add_argument('--expected-projection-groups', type=int,
                        choices=(1, 2, 3, 4, 5, 6), default=1)
    parser.add_argument('--expected-shared-layers', type=int,
                        choices=(1, 2), default=1)
    args = parser.parse_args()
    if not args.bundle.is_dir():
        report = {'status': 'NOT_TESTED', 'bundle': str(args.bundle),
                  'reason': 'bundle directory unavailable'}
    else:
        try:
            report = certify(
                args.bundle.resolve(), swap_row_order=args.swap_row_order,
                expected_projection_groups=args.expected_projection_groups,
                expected_shared_layers=args.expected_shared_layers)
        except (OSError, ValueError, AssertionError, RuntimeError) as error:
            report = {'status': 'FAIL', 'bundle': str(args.bundle),
                      'error': f'{type(error).__name__}: {error}'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return {'PASS': 0, 'FAIL': 1, 'NOT_TESTED': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
