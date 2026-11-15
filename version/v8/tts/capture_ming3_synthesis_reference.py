#!/usr/bin/env python3
"""Offline publisher CPU reference; this is not a CKE synthesis runner."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SOURCE_REVISION='a39b80660bdc5a4394d5e1d41af6dbfd9c443a54'
CHECKPOINT_REVISION='7247f4c94a5b2e987404ed28a5cfec41b1a4ca61'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publisher-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-decode-steps',type=int,default=128)
    p.add_argument('--decoder-fp32-replay',action='store_true',help='also capture separate recast-FP32 decoder references')
    args=p.parse_args()
    if not 1<=args.threads<=64 or not 4<=args.max_decode_steps<=512:
        p.error('threads must be 1..64; decode steps must be 4..512')
    root=args.publisher_root.resolve(); checkpoint=args.checkpoint.resolve()
    rev=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if rev!=SOURCE_REVISION: p.error('publisher source revision mismatch')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip():
        p.error('publisher tracked source must be clean')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'scope':'publisher CPU reference, not CKE synthesis or numerical certification',
        'source_revision':rev,'checkpoint_revision_expected':CHECKPOINT_REVISION,
        'checkpoint_identity_policy':'actual local file hashes; revision label alone is not verified',
        'threads':args.threads,'seed':731,'max_decode_steps':args.max_decode_steps,
        'text':'Hello. This is Ming speech.','voice':'UK_George',
        'temperature':0.0,'cfg':2.0,'sigma':0.25,'status':'INCOMPLETE'}
    report_path=args.output/'report.json'
    def save(): report_path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        report['checkpoint_files']={path.name:digest(path) for path in sorted(checkpoint.iterdir()) if path.is_file() and not path.name.endswith('.part')}
        sys.path.insert(0,str(root))
        import torch
        import transformers
        import soundfile
        from ming_tts import MingTTS
        report['torch_version']=torch.__version__;report['transformers_version']=transformers.__version__
        torch.set_num_threads(args.threads);torch.manual_seed(731)
        start=time.monotonic()
        model=MingTTS(str(checkpoint),device='cpu',text_normalization=False)
        report['load_seconds']=time.monotonic()-start
        seen=set()
        def hook(name):
            def capture(module,inputs,output):
                if name in seen:return
                seen.add(name)
                def cpu(value):
                    if isinstance(value,torch.Tensor):return value.detach().cpu()
                    if isinstance(value,(tuple,list)):return tuple(cpu(x) for x in value)
                    return value
                payload={'inputs':cpu(inputs),'output':cpu(output)}
                if name=='dit_first_block':
                    import copy
                    payload['weights']={key:cpu(value) for key,value in module.state_dict().items()}
                    fp32=copy.deepcopy(module).float().eval()
                    for child in fp32.modules():child._forward_hooks.clear()
                    def floating(value):
                        if isinstance(value,torch.Tensor):return value.float()
                        if isinstance(value,(tuple,list)):return tuple(floating(x) for x in value)
                        return value
                    with torch.inference_mode():payload['fp32_replay_output']=cpu(fp32(*floating(inputs)))
                    payload['reference_scope']='same captured inputs and weights recast to FP32; separate from BF16 trajectory'
                torch.save(payload,args.output/(name+'.pt'))
            return capture
        model.model.flowloss.cfm.model.blocks[0].register_forward_hook(hook('dit_first_block'))
        model.model.audio.decoder.fc1.register_forward_hook(hook('vae_fc1'))
        model.model.audio.decoder.head.out.register_forward_hook(hook('vae_head_projection'))
        steps=[]
        original=model.model.sample
        def monitored_sample(*a,**kw):
            for latent,ended in original(*a,**kw):
                steps.append(bool(ended));yield latent,ended
        model.model.sample=monitored_sample
        start=time.monotonic()
        wav=model.synthesize(task='IP',text=report['text'],language_id='en',controls={'ip':report['voice']},
            raw_text=True,max_decode_steps=args.max_decode_steps,trim_silence=False)
        report['synthesis_seconds']=time.monotonic()-start
        if not wav.numel() or not bool(torch.isfinite(wav).all()): raise ValueError('empty or nonfinite waveform')
        soundfile.write(str(args.output/'publisher.wav'),wav.float().numpy().reshape(-1),model.sample_rate)
        report.update(samples=wav.numel(),sample_rate=model.sample_rate,steps=len(steps),
            normal_stop=bool(steps and steps[-1]),status='PASS' if steps and steps[-1] else 'INCOMPLETE')
        # Preserve actual spectral conversion and the BF16 window, not a
        # recomputed FP32 approximation. These remain oracle-fed native inputs.
        import numpy as np
        head=torch.load(args.output/'vae_head_projection.pt',weights_only=True)['output']
        magnitude,phase=head.transpose(1,2).chunk(2,dim=1)
        magnitude=magnitude.exp().clamp(max=100)
        spectrum=magnitude*(phase.cos()+1j*phase.sin())
        np.savez(args.output/'waveform_operands.npz',
            real=spectrum.real[0].T.contiguous().numpy(),
            imag=spectrum.imag[0].T.contiguous().numpy(),
            window=model.model.audio.decoder.head.istft.window.float().cpu().numpy(),
            expected=wav.float().cpu().numpy().reshape(-1))
        if args.decoder_fp32_replay:
            decoder=model.model.audio.decoder.float().eval()
            latents=torch.load(args.output/'vae_fc1.pt',weights_only=True)['inputs'][0].float()
            captures={}
            def keep(name):
                def retain(module,inputs,output):
                    if hasattr(output,'last_hidden_state'):output=output.last_hidden_state
                    captures[name]=output.detach().cpu().numpy()[0]
                return retain
            handles=[decoder.fc1.register_forward_hook(keep('projection')),
                     decoder.decoder.register_forward_hook(keep('final_norm')),
                     decoder.head.out.register_forward_hook(keep('spectral_projection'))]
            lengths=sorted(set([min(2,latents.shape[1]),min(17,latents.shape[1]),latents.shape[1]]))
            try:
                for length in lengths:
                    with torch.inference_mode():
                        audio,_,_=decoder.low_level_reconstruct(latents[:,:length],stream_state=(None,None,None))
                    np.savez(args.output/f'decoder-fp32-{length}.npz',latent=latents[0,:length].numpy(),
                        window=decoder.head.istft.window.detach().cpu().numpy(),output=audio.cpu().numpy().reshape(-1),**captures)
            finally:
                for handle in handles:handle.remove()
            report['decoder_fp32_replay']={'scope':'Recast FP32 decoder from publisher latents; NOT BF16 parity',
                'lengths':lengths,'attention_implementation':decoder.decoder.config._attn_implementation,
                'sliding_window':decoder.decoder.config.sliding_window}
        report['outputs']={path.name:digest(path) for path in args.output.iterdir() if path.name!='report.json'}
    except Exception as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));save();raise
    save()
    if report['status']!='PASS': raise SystemExit('Reference reached budget without normal stop')

if __name__=='__main__':main()
