#!/usr/bin/env python3
"""Asset-backed native-session checkpoint continuation and recovery probe.

The one-token raw suffix follows a successful forward boundary. This checks
session control flow plus selected-model logits/state, not arbitrary chunking.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(source))
from server.session_v8 import _Config, _GenerateRequest, _GenerateResult
from certify_sequence_checkpoint_v8 import configure, declared_state_regions, signature, sha_file

RAW_PROMPT = 1 << 0
CONTINUE_STATE = 1 << 3
REJECT_TRUNCATION = 1 << 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--session-library', type=Path, required=True)
    parser.add_argument('--session-build-commit', required=True,
                        help='commit used to build the supplied native session library')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--context-length', type=int, required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--prefix', default='Hello')
    parser.add_argument('--other', default='Different')
    parser.add_argument('--suffix', default=' world')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    session_library = args.session_library.resolve()
    if args.context_length <= 0 or args.threads <= 0 or not all(
            (args.prefix, args.other, args.suffix)):
        parser.error('positive context/threads and nonempty text fixtures required')
    native = C.CDLL(str(session_library))
    native.ck_session_v8_open.argtypes = [C.POINTER(_Config), C.POINTER(C.c_void_p)]
    native.ck_session_v8_open.restype = C.c_int
    native.ck_session_v8_close.argtypes = [C.c_void_p]
    native.ck_session_v8_checkpoint_requirements.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    native.ck_session_v8_checkpoint_requirements.restype = C.c_int
    native.ck_session_v8_checkpoint_save.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    native.ck_session_v8_checkpoint_save.restype = C.c_int
    native.ck_session_v8_checkpoint_restore.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    native.ck_session_v8_checkpoint_restore.restype = C.c_int
    native.ck_session_v8_generate.argtypes = [C.c_void_p, C.POINTER(_GenerateRequest), C.c_void_p, C.c_void_p, C.POINTER(_GenerateResult)]
    native.ck_session_v8_generate.restype = C.c_int
    native.ck_session_v8_last_error.argtypes = [C.c_void_p]
    native.ck_session_v8_last_error.restype = C.c_char_p

    config = _Config(struct_size=C.sizeof(_Config), abi_version=1,
                     model_library_path=str(bundle / 'libmodel.so').encode(),
                     weights_path=str(bundle / 'weights.bump').encode(),
                     manifest_path=str(bundle / 'weights_manifest.map').encode(),
                     context_length=args.context_length, num_threads=args.threads,
                     required_capabilities=1 << 18)
    session = C.c_void_p()
    rc = native.ck_session_v8_open(C.byref(config), C.byref(session))
    if rc or not session.value:
        raise RuntimeError(f'session open failed: {rc}')
    try:
        model = C.CDLL(str(bundle / 'libmodel.so'), mode=C.RTLD_GLOBAL)
        loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if len(line.split()) >= 6}
        expected_libraries = {str(session_library), str(bundle / 'libmodel.so'),
                              str(bundle / 'libckernel_engine.so')}
        if not expected_libraries <= loaded:
            raise RuntimeError('loaded library identity mismatch')
        configure(model)
        model.ck_model_get_logits.restype = C.c_void_p
        model.ck_model_get_logits_stride.restype = C.c_int
        required = declared_state_regions(json.loads((bundle / 'layout_decode.json').read_text()))
        vocab = model.ck_model_get_vocab_size()
        def sig():
            addr = model.ck_model_get_logits()
            stride = model.ck_model_get_logits_stride()
            active = model.ck_model_get_active_tokens()
            if not addr or vocab <= 0 or active <= 0:
                raise RuntimeError('invalid logits extent')
            if stride > 0:
                addr += (active - 1) * stride * C.sizeof(C.c_float)
            logits = (C.c_float * vocab).from_address(addr)
            return signature(model, logits, required)
        def generate(text, flags):
            request = _GenerateRequest(struct_size=C.sizeof(_GenerateRequest), abi_version=1,
                                       user_text=text.encode(), max_tokens=1, temperature=0,
                                       top_p=1, flags=flags)
            result = _GenerateResult(struct_size=C.sizeof(_GenerateResult), abi_version=1)
            rc = native.ck_session_v8_generate(session, C.byref(request), None, None, C.byref(result))
            if rc:
                raise RuntimeError(f'generate failed {rc}: {native.ck_session_v8_last_error(session)}')
            if result.stop_reason != 2:
                raise RuntimeError(f'unexpected stop reason {result.stop_reason}')
            return {'prompt_tokens': result.prompt_tokens, 'prefill_ms': result.prefill_time_ms,
                    'state': sig()}
        raw = RAW_PROMPT | REJECT_TRUNCATION
        cont = raw | CONTINUE_STATE
        first = generate(args.prefix, raw)
        size, align = C.c_size_t(), C.c_size_t()
        if native.ck_session_v8_checkpoint_requirements(session, C.byref(size), C.byref(align)):
            raise RuntimeError('checkpoint requirements failed')
        storage = C.create_string_buffer(size.value + align.value)
        snapshot = (C.addressof(storage) + align.value - 1) & ~(align.value - 1)
        if native.ck_session_v8_checkpoint_save(session, snapshot, size.value):
            raise RuntimeError('checkpoint save failed')
        baseline = generate(args.suffix, cont)
        other = generate(args.other, raw)
        if native.ck_session_v8_checkpoint_restore(session, snapshot, size.value):
            raise RuntimeError('checkpoint restore failed')
        restored = generate(args.suffix, cont)
        after = sig()
        original = C.c_uint8.from_address(snapshot).value
        C.c_uint8.from_address(snapshot).value ^= 1
        reject_rc = native.ck_session_v8_checkpoint_restore(session, snapshot, size.value)
        rejected_unchanged = after == sig()
        C.c_uint8.from_address(snapshot).value = original
        report = {
            'schema': 'cke.session_checkpoint_parity.v1',
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
            'session_build_commit': args.session_build_commit,
            'probe_sha256': sha_file(Path(__file__)),
            'inputs': {'prefix': args.prefix, 'other': args.other, 'suffix': args.suffix},
            'bundle': str(bundle),
            'artifact_sha256': {name: sha_file(bundle / name) for name in
                ('libmodel.so', 'libckernel_engine.so', 'model_v8.c',
                 'weights_manifest.json', 'weights_manifest.map', 'config.json', 'weights.bump')},
            'session_library_sha256': sha_file(session_library),
            'loaded_libraries': sorted(expected_libraries),
            'threads': {'CK_NUM_THREADS': os.environ.get('CK_NUM_THREADS'),
                        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                        'session_num_threads': args.threads},
            'cpu': next((line.split(':', 1)[1].strip()
                         for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name')), None),
            'required_state': required,
            'checkpoint_bytes': size.value,
            'checkpoint_alignment': align.value,
            'context_capacity': args.context_length,
            'prefill_policy': json.loads((bundle / 'config.json').read_text()).get('prefill_policy'),
            'first': first,
            'other': other,
            'baseline': baseline,
            'restored': restored,
            'exact': baseline['state'] == restored['state'],
            'corrupt_restore_rejected': reject_rc != 0 and rejected_unchanged,
        }
        report['verdict'] = 'PASS' if (
            report['exact'] and report['corrupt_restore_rejected'] and
            first['prompt_tokens'] > 0 and other['prompt_tokens'] > 0 and
            baseline['prompt_tokens'] == restored['prompt_tokens'] == 1
        ) else 'FAIL'
        output = args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(report['verdict'], 'prompt_tokens', first['prompt_tokens'], baseline['prompt_tokens'], flush=True)
    finally:
        native.ck_session_v8_close(session)

    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
