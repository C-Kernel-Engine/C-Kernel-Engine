#!/usr/bin/env python3
"""Replay complete FP32 generated audio decoder from retained publisher latents."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_audio_decoder_circuit import build_circuit
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes as weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph as compile_block
from version.v8.tts.capture_ming3_synthesis_reference import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    report={'status':'INCOMPLETE','scope':'FP32 native latents-to-waveform, not text-to-speech or publisher BF16 parity'}
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        with np.load(a.reference,allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
        x=data['latent']
        if x.ndim!=2 or x.shape[1]!=64 or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('invalid latent fixture')
        graph=build_circuit(frames=len(x))
        if data['window'].shape!=(3528,) or not np.isfinite(data['window']).all():raise ValueError('invalid window')
        report.update(reference_sha256=digest(a.reference),checkpoint_sha256=digest(a.checkpoint),frames=len(x))
        weights={}
        with safe_open(str(a.checkpoint),framework='pt') as source:
            for key in weight_shapes(graph):weights[key]=source.get_tensor('audio.decoder.'+key).float().numpy().copy()
        fn,arena,view,loaded=compile_block(a.output,graph,weights,openmp=True,c_standard='gnu11',provider_warnings='report')
        del weights
        view('latent')[:]=x;view('window')[:]=data['window']
        start=time.monotonic();status=fn(arena,len(arena));elapsed=time.monotonic()-start
        if status:raise RuntimeError(f'native status {status}')
        errors={}
        for key in ('projection','final_norm','spectral_projection','output'):
            actual=view(key).copy();expected=data[key]
            if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('invalid comparison '+key)
            diff=actual.astype('f8')-expected.astype('f8')
            errors[key]={'max_error':float(abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff**2)))}
        waveform=view('output').copy();np.save(a.output/'waveform.npy',waveform)
        import wave
        with wave.open(str(a.output/'native-from-reference-latents.wav'),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(44100)
            wav.writeframes((np.clip(waveform,-1,1)*32767).astype('<i2').tobytes())
        report.update(status='MEASURED_NOT_CERTIFIED',seconds=elapsed,samples=len(waveform),arena_bytes=len(arena),errors=errors,
            generated_library_sha256=digest(a.output/'generated.so'),build_identity_sha256=digest(a.output/'build-identity.json'))
    except Exception as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));save();raise
    save();print(json.dumps(report))
if __name__=='__main__':main()
