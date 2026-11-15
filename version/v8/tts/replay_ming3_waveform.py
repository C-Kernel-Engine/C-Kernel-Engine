#!/usr/bin/env python3
"""Run generated C on retained oracle spectra; NOT end-to-end text synthesis."""
import argparse,ctypes,hashlib,json,sys,time,wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_waveform_circuit import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operands',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(a.operands,allow_pickle=False)
    real=data['real'];imag=data['imag'];window=data['window']
    if real.ndim!=2 or imag.shape!=real.shape or window.shape!=(3528,) or real.shape[1]!=1765:
        p.error('expected frame-major Ming spectra with n_fft=3528')
    graph=build_circuit(len(real));path=a.output/'circuit.json';path.write_text(json.dumps(graph,indent=2)+'\n')
    source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':2,
        'num_heads':1,'num_kv_heads':1,'head_dim':2,'intermediate_size':2,
        'context_length':len(real),'max_seq_len':len(real),'vocab_size':1},'entries':[],
        'quant_summary':{},'template':graph}
    layout,calls,library,loaded,fn=compile_native_graph(a.output,source,path)
    size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))()
    arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
    buffers={b['name']:b for b in layout['memory']['activations']['buffers']}
    def view(name):return np.ndarray(tuple(graph['activation_buffers'][name]['shape']),np.float32,buffer=arena,offset=buffers[name]['abs_offset'])
    for name,value in [('real',real),('imag',imag),('window',window)]:view(name)[:]=value
    start=time.monotonic();status=fn(arena,size);elapsed=time.monotonic()-start
    if status:raise RuntimeError(f'native returned {status}')
    output=view('output').copy();np.save(a.output/'waveform.npy',output)
    pcm=np.rint(np.clip(output,-1,1)*32767).astype('<i2')
    with wave.open(str(a.output/'native-from-oracle-spectrum.wav'),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(44100);wav.writeframes(pcm.tobytes())
    report={'scope':'native generated waveform from oracle spectra; NOT full synthesis',
        'frames':len(real),'samples':len(output),'seconds':elapsed,'arena_bytes':size,
        'operands_sha256':hashlib.sha256(a.operands.read_bytes()).hexdigest(),
        'library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((ROOT/'src/kernels/audio_istft_complex_window_checked.c').read_bytes()).hexdigest(),
        'finite':bool(np.isfinite(output).all()),'numerical_verdict':'NOT_TESTED'}
    if 'expected' in data:
        expected=data['expected']
        if expected.shape!=output.shape or not np.isfinite(expected).all():raise ValueError('invalid reference')
        error=output.astype('f8')-expected.astype('f8')
        report.update(max_error=float(abs(error).max()),rmse=float(np.sqrt(np.mean(error**2))),
            numerical_verdict='MEASURED_NOT_CERTIFIED')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
