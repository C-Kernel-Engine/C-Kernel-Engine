"""Test-only ordinary compiler/host setup for circuit-declared checked graphs."""
import contextlib
import ctypes
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
import build_ir_v8


def compile_native_graph(root, source, circuit, extra_exports=(), mode="prefill"):
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode=mode)
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, mode)
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode=mode, context_len=source['config']['context_length'])
        lower2 = build_ir_v8.generate_ir_lower_2(lower1, layout, source, registry, mode=mode)
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode=mode)
    (root / 'layout.json').write_text(json.dumps(layout))
    (root / 'call.json').write_text(json.dumps(call_ir))
    generated = root / 'generated.c'
    codegen = subprocess.run([sys.executable, str(ROOT / 'version/v8/scripts/codegen_v8.py'),
                    '--ir', str(root / 'call.json'), '--layout', str(root / 'layout.json'),
                    '--output', str(generated)], capture_output=True, text=True)
    if codegen.returncode:
        raise RuntimeError('native graph generation failed:\n' + codegen.stdout + codegen.stderr)
    function = source['template']['native_entry']['function']
    exports = root / 'exports.map'
    exports.write_text('{ global: ' + '; '.join((function, *extra_exports)) +
                       '; local: *; };\n')
    providers = {provider["id"]: provider for provider in registry["kernels"]}
    sources = set()
    for op in call_ir['operations']:
        provider = providers[op['call_abi']['kernel_id']]
        sources.update(provider['impl']['sources'])
    # Generated code remains ISO C checked. Legacy dependencies include POSIX
    # dlsym conversions; compile/link them under their established conventions.
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
                    '-fsyntax-only', str(generated), '-I', str(ROOT / 'include')],
                   check=True, capture_output=True, text=True)
    library = root / 'generated.so'
    result = subprocess.run([
        'cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
        '-ffp-contract=off', '-ffunction-sections', '-fdata-sections',
        '-Wl,--gc-sections', '-Wl,--no-undefined', '-Wl,--version-script=' + str(exports),
        str(generated), *[str(ROOT / name) for name in sorted(sources)],
        '-I', str(ROOT / 'include'), '-lm', '-lpthread', '-ldl', '-o', str(library),
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('native graph compilation failed:\n' + result.stderr)
    identity = {
        'compiler': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
        'compile_command': result.args,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources)},
        'generated_source_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'embedded_runtime_build_identity': 'NOT_VERIFIED',
    }
    (root / 'build-identity.json').write_text(json.dumps(identity, indent=2))
    loaded = ctypes.CDLL(str(library))
    fn = getattr(loaded, function)
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return layout, call_ir, library, loaded, fn
