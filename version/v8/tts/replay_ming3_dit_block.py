#!/usr/bin/env python3
"""Generated first DiT block replay, explicitly FP32 rather than BF16 parity."""
import argparse,ctypes,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit,weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

def compile_block(folder,graph,weights):
    entries=[];offset=4096
    for name,shape in weight_shapes(graph).items():
        w=weights[name]
        if list(w.shape)!=shape or w.dtype!=np.float32:raise ValueError(name)
        offset=(offset+63)&~63
        entries.append({'name':name,'shape':shape,'dtype':'F32','size':w.nbytes,'file_offset':offset})
        offset+=w.nbytes
    c=graph['contract']['runtime_invariants'];width=c['width'];tokens=c['tokens'];heads=c['heads']
    source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':width,
        'num_heads':heads,'num_kv_heads':heads,'head_dim':width//heads,'intermediate_size':width*c['mult'],
        'context_length':tokens,'max_seq_len':tokens,'vocab_size':1},'entries':entries,'quant_summary':{},'template':graph}
    path=folder/'circuit.json';path.write_text(json.dumps(graph))
    layout,calls,library,loaded,fn=compile_native_graph(folder,source,path)
    size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))()
    arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
    for item in layout['memory']['weights']['entries']:
        w=weights[item['name']];start=item['abs_offset'];arena[start:start+w.nbytes]=w.tobytes()
    buffers={b['name']:b for b in layout['memory']['activations']['buffers']}
    def view(name):return np.ndarray(tuple(graph['activation_buffers'][name]['shape']),np.float32,buffer=arena,offset=buffers[name]['abs_offset'])
    return fn,arena,view,loaded


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
