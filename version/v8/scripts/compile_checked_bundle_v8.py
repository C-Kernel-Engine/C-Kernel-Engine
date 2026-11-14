"""Build a bounded checked graph through the ordinary v8 compiler and providers."""
import contextlib
import ctypes
import io
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
import build_ir_v8


def compile_native_graph(root, source, circuit, extra_exports=(), mode="prefill", *,
                         runtime_sources=(), openmp=False, fp_contract="off",
                         provider_warnings="error", c_standard="c11"):
    """Compile resolved providers, with explicit native-shell support and policy.

    Neural sources come exclusively from resolved maps. Extra sources are limited
    to src/runtime, for native host shells, and are separately recorded. Generated
    code always passes strict C11 checks, even when legacy providers report warnings.
    """
    root = Path(root)
    if type(openmp) is not bool or fp_contract not in ("off", "fast") or provider_warnings not in ("error", "report") or c_standard not in ("c11", "gnu11"):
        raise ValueError("invalid native build policy")
    support = set()
    for name in runtime_sources:
        path = (ROOT / name).resolve()
        if not path.is_relative_to((ROOT / "src/runtime").resolve()) or path.suffix != ".c" or not path.is_file():
            raise ValueError("extra native support must be a declared src/runtime C source")
        support.add(str(path.relative_to(ROOT)))
    entry = source['template']['native_entry']
    exported = {entry['function'], *extra_exports,
                *(segment['function'] for segment in entry.get('segments', []))}
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in exported):
        raise ValueError("invalid native export name")
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode=mode)
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, mode)
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode=mode, context_len=source['config']['context_length'])
        lower2 = build_ir_v8.generate_ir_lower_2(lower1, layout, source, registry, mode=mode)
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode=mode)
    if call_ir.get('errors'):
        raise ValueError("checked graph lowering failed: " + str(call_ir['errors']))
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
    exports.write_text('{ global: ' + '; '.join(sorted(exported)) +
                       '; local: *; };\n')
    providers = {provider["id"]: provider for provider in registry["kernels"]}
    sources = set()
    selected = {}
    for op in call_ir['operations']:
        provider = providers[op['call_abi']['kernel_id']]
        sources.update(provider['impl']['sources'])
        selected[provider['id']] = hashlib.sha256(json.dumps(provider, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    # Generated code remains ISO C checked. Legacy dependencies include POSIX
    # dlsym conversions; compile/link them under their established conventions.
    syntax = subprocess.run(
        ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fsyntax-only', str(generated), '-I', str(ROOT / 'include')],
        capture_output=True, text=True)
    (root / 'generated-diagnostics.txt').write_text(syntax.stdout + syntax.stderr)
    if syntax.returncode:
        raise RuntimeError('generated graph syntax validation failed:\n' + syntax.stderr)
    library = root / 'generated.so'
    result = subprocess.run([
        'cc', '-std=' + c_standard, '-O2', '-Wall', '-Wextra',
        *(['-Werror'] if provider_warnings == 'error' else []), '-shared', '-fPIC',
        '-ffp-contract=' + fp_contract, *(['-fopenmp'] if openmp else []), '-ffunction-sections', '-fdata-sections',
        '-Wl,--gc-sections', '-Wl,--no-undefined', '-Wl,--version-script=' + str(exports),
        str(generated), *[str(ROOT / name) for name in sorted(sources | support)],
        '-I', str(ROOT / 'include'), '-lm', '-lpthread', '-ldl', '-o', str(library),
    ], capture_output=True, text=True)
    (root / 'compiler-diagnostics.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('native graph compilation failed:\n' + result.stderr)
    identity = {
        'build_policy': dict(openmp=openmp, fp_contract=fp_contract, provider_warnings=provider_warnings, c_standard=c_standard),
        'selected_provider_sha256': selected,
        'runtime_support_sources': sorted(support),
        'exports': sorted(exported),
        'call_ir_sha256': hashlib.sha256((root / 'call.json').read_bytes()).hexdigest(),
        'layout_sha256': hashlib.sha256((root / 'layout.json').read_bytes()).hexdigest(),
        'compiler_diagnostics_sha256': hashlib.sha256((root / 'compiler-diagnostics.txt').read_bytes()).hexdigest(),
        'compiler_warning_lines': sum('warning:' in line for line in result.stderr.splitlines()),
        'compiler': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
        'compile_command': result.args,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources | support)},
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
