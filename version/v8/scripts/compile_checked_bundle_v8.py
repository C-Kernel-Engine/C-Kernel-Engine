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
import shutil
import tempfile
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
import build_ir_v8
from checked_bundle_evidence_v8 import (
    IDENTITY_SYMBOL, WARNING_POLICY, check_provider_warnings, digest_json,
    load_verified_checked_bundle,
)


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
    if IDENTITY_SYMBOL in exported:
        raise ValueError("native entry collides with reserved build-identity export")
    exported.add(IDENTITY_SYMBOL)
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
    compiler_env = dict(os.environ, LC_ALL="C", LANG="C")
    syntax = subprocess.run(
        ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fsyntax-only', str(generated), '-I', str(ROOT / 'include')],
        capture_output=True, text=True, env=compiler_env)
    (root / 'generated-diagnostics.txt').write_text(syntax.stdout + syntax.stderr)
    if syntax.returncode:
        raise RuntimeError('generated graph syntax validation failed:\n' + syntax.stderr)
    compiler = subprocess.check_output(['cc', '--version'], text=True, env=compiler_env).splitlines()[0]
    policy = dict(openmp=openmp, fp_contract=fp_contract, provider_warnings=provider_warnings, c_standard=c_standard)
    identity_inputs = dict(
        schema="cke.checked-native-build.v1", compiler=compiler, build_policy=policy,
        compiler_executable_sha256=hashlib.sha256(Path(shutil.which("cc")).resolve().read_bytes()).hexdigest(),
        builder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_provider_sha256=selected,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources | support)},
        header_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "include").rglob("*.h"))},
        generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        call_ir_sha256=hashlib.sha256((root / 'call.json').read_bytes()).hexdigest(),
        layout_sha256=hashlib.sha256((root / 'layout.json').read_bytes()).hexdigest(),
        warning_policy_sha256=hashlib.sha256(WARNING_POLICY.read_bytes()).hexdigest(),
        exports=sorted(exported),
    )
    build_digest = digest_json(identity_inputs)
    identity_source = root / 'build-identity.c'
    identity_source.write_text('const char *' + IDENTITY_SYMBOL + '(void) { return "' + build_digest + '"; }\n')
    library = root / 'generated.so'
    staged_fd, staged_name = tempfile.mkstemp(prefix='checked-build-', suffix='.so', dir=root)
    os.close(staged_fd)
    staged_library = Path(staged_name)
    try:
        result = subprocess.run([
            'cc', '-std=' + c_standard, '-O2', '-Wall', '-Wextra',
            *(['-Werror'] if provider_warnings == 'error' else []), '-shared', '-fPIC',
            '-ffp-contract=' + fp_contract, *(['-fopenmp'] if openmp else []), '-ffunction-sections', '-fdata-sections',
            '-Wl,--gc-sections', '-Wl,--no-undefined', '-Wl,--version-script=' + str(exports),
            str(generated), str(identity_source), *[str(ROOT / name) for name in sorted(sources | support)],
            '-I', str(ROOT / 'include'), '-lm', '-lpthread', '-ldl', '-o', str(staged_library),
        ], capture_output=True, text=True, env=compiler_env)
    except BaseException:
        staged_library.unlink(missing_ok=True)
        raise
    (root / 'compiler-diagnostics.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        staged_library.unlink(missing_ok=True)
        raise RuntimeError('native graph compilation failed:\n' + result.stderr)
    try:
        accepted_warnings = check_provider_warnings(result.stdout + "\n" + result.stderr)
        os.replace(staged_library, library)
    finally:
        staged_library.unlink(missing_ok=True)
    identity = {
        'identity_inputs': identity_inputs,
        'build_identity_sha256': build_digest,
        'accepted_provider_warnings': accepted_warnings,
        'build_policy': dict(openmp=openmp, fp_contract=fp_contract, provider_warnings=provider_warnings, c_standard=c_standard),
        'selected_provider_sha256': selected,
        'runtime_support_sources': sorted(support),
        'exports': sorted(exported),
        'call_ir_sha256': hashlib.sha256((root / 'call.json').read_bytes()).hexdigest(),
        'layout_sha256': hashlib.sha256((root / 'layout.json').read_bytes()).hexdigest(),
        'compiler_diagnostics_sha256': hashlib.sha256((root / 'compiler-diagnostics.txt').read_bytes()).hexdigest(),
        'compiler_warning_lines': sum('warning:' in line for line in result.stderr.splitlines()),
        'compiler': subprocess.check_output(['cc', '--version'], text=True, env=compiler_env).splitlines()[0],
        'compile_command': result.args,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources | support)},
        'generated_source_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'embedded_runtime_build_identity': 'NOT_VERIFIED',
    }
    (root / 'build-identity.json').write_text(json.dumps(identity, indent=2))
    loaded = load_verified_checked_bundle(library, root / 'build-identity.json',
        hashlib.sha256((root / 'build-identity.json').read_bytes()).hexdigest())
    identity['embedded_runtime_build_identity'] = 'VERIFIED'
    (root / 'build-identity.json').write_text(json.dumps(identity, indent=2))
    fn = getattr(loaded, function)
    scalar_types = {"float": ctypes.c_float, "int32_t": ctypes.c_int32,
                    "size_t": ctypes.c_size_t, "uint8_t": ctypes.c_uint8,
                    "int": ctypes.c_int}
    def parameter_type(parameter):
        declared = parameter["c_type"].replace("const ", "").strip()
        pointers = declared.count("*")
        base = declared.replace("*", "").strip()
        if base == "void" and pointers == 1:
            return ctypes.c_void_p
        if base not in scalar_types:
            raise ValueError(f"unsupported native entry parameter type: {declared}")
        result = scalar_types[base]
        for _ in range(pointers):
            result = ctypes.POINTER(result)
        return result
    fn.argtypes = [parameter_type(p) for p in source["template"]["native_entry"]["params"]]
    fn.restype = ctypes.c_int
    return layout, call_ir, library, loaded, fn
