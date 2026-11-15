#!/usr/bin/env python3
"""Generated first DiT block replay, explicitly FP32 rather than BF16 parity."""
import argparse,ctypes,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit,weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph as compile_block

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    data=torch.load(a.capture,weights_only=True,map_location='cpu')
    x=data['inputs'][0].float().numpy();freq,scale=data['inputs'][2]
    if x.ndim!=3 or x.shape[-1]!=1024 or scale!=1.:raise ValueError('unsupported captured block shape or rotary scale')
    graph=build_circuit(tokens=x.shape[1]);weights={k:v.float().numpy().copy() for k,v in data['weights'].items()}
    if set(weights)!=set(weight_shapes(graph)):raise ValueError('unexpected block weights')
    expected=data['fp32_replay_output'].float().numpy()
    if expected.shape!=x.shape or not np.isfinite(expected).all():raise ValueError('invalid reference')
    cosine=freq.float().cos()[0,:,::2].numpy();sine=freq.float().sin()[0,:,::2].numpy()
    a.output.mkdir(parents=True,exist_ok=False)
    fn,arena,view,loaded=compile_block(a.output,graph,weights);results=[]
    for branch in range(len(x)):
        view('input')[:]=x[branch];view('cosine')[:]=cosine;view('sine')[:]=sine
        start=time.monotonic();status=fn(arena,len(arena));elapsed=time.monotonic()-start
        if status:raise RuntimeError(f'native returned {status}')
        output=view('output').copy();np.save(a.output/f'branch-{branch}.npy',output)
        error=output.astype('f8')-expected[branch].astype('f8')
        results.append({'branch':branch,'seconds':elapsed,'max_error':float(abs(error).max()),'rmse':float(np.sqrt(np.mean(error**2)))})
    report={'scope':'real-weight generated FP32 block vs recast FP32 publisher block; NOT BF16 parity',
        'numerical_verdict':'MEASURED_NOT_CERTIFIED','capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        'library_sha256':hashlib.sha256((a.output/'generated.so').read_bytes()).hexdigest(),
        'results':results}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
