"""A bridge bundle must load in a fresh process without global tokenizer preload."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_bridge_bundle_relocation_loads_packaged_tokenizer(tmp_path):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'version/v8/scripts'))
    from run_multimodal_bridge_v8 import _compile_generated_model
    build=subprocess.run(['make','--no-print-directory','build/libckernel_engine.so','build/libckernel_tokenizer.so'],
        cwd=root,capture_output=True,text=True,timeout=180)
    assert build.returncode==0,build.stdout+build.stderr
    source=tmp_path/'model.c'
    source.write_text('extern void ck_true_bpe_decode(void);\nvoid *probe(void) { return (void *)ck_true_bpe_decode; }\n')
    model=tmp_path/'model.so';_compile_generated_model(source,model)
    relocated=tmp_path/'relocated';relocated.mkdir()
    for name in ['model.so','libckernel_engine.so','libckernel_tokenizer.so']:
        shutil.copy2(tmp_path/name,relocated/name)
    child='''import ctypes,json,sys
from pathlib import Path
lib=ctypes.CDLL(sys.argv[1]);lib.probe.restype=ctypes.c_void_p
assert lib.probe()
paths={str(Path(line.split(maxsplit=5)[5]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines() if line.endswith('/libckernel_tokenizer.so')}
print(json.dumps(sorted(paths)))
'''
    env={k:v for k,v in os.environ.items() if k not in {'LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'}}
    result=subprocess.run([sys.executable,'-c',child,str(relocated/'model.so')],
        cwd=relocated,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==[str(relocated/'libckernel_tokenizer.so')]
    stamp=json.loads(model.with_suffix('.so.build.json').read_text())
    assert stamp['tokenizer_dependency']['sha256']==hashlib.sha256((relocated/'libckernel_tokenizer.so').read_bytes()).hexdigest()
