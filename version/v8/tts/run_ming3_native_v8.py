#!/usr/bin/env python3
"""Experimental native Ming FP32 text-to-WAV; no publisher neural fallback.

Python prepares tokens, maps weights, invokes one native generation loop and one
native decoder, and writes WAV. Full-prefix and checked KV-append policies are
explicitly selected. This is
not publisher BF16/RNG equivalence, streaming synthesis, or a production service.
"""
from __future__ import annotations
import argparse,ctypes as C,json,os,subprocess,sys,threading,time,wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_backbone_circuit import build_circuit as backbone
from version.v8.tts.build_ming3_aggregator_circuit import build_circuit as aggregator
from version.v8.tts.build_ming3_dit_circuit import build_circuit as dit,constant_weights
from version.v8.tts.build_ming3_runtime_audio_decoder_circuit import build_circuit as decoder
from version.v8.tts.build_ming3_sampler_update_circuit import build_circuit as update
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes as weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph as compile_block
from version.v8.tts.native_guided_flow import CANCEL
from version.v8.tts.native_latent_loop import LatentLoop,bind_latent_request
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.latent_execution_contract_v8 import resolve_latent_execution,native_shell_spec
from version.v8.tts.ming3_execution_profile import load_profile
from version.v8.tts.capture_ming3_synthesis_reference import SOURCE_REVISION,digest

PINNED_WEIGHTS_SHA256='45c4d78d8ad6b1d942e3da86ba035fece02de569f7b2ca7e9d121dfaf18d674c'
PINNED_TOKENIZER_SHA256='87d54ca3b44ea2f822ba39a0a45cb1b61644bd6606d9d3536dea75367bffb06d'

class IncompleteSynthesis(RuntimeError):pass
class CancelledSynthesis(RuntimeError):pass


def prepare_tokens(text,checkpoint,publisher_root):
    """Pinned publisher's pure prompt contract; no model import or execution."""
    rev=subprocess.check_output(['git','-C',str(publisher_root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(publisher_root),'status','--porcelain','--untracked-files=no'],text=True)
    if rev!=SOURCE_REVISION or dirty:raise ValueError('publisher prompt source must match the clean pinned revision')
    if not isinstance(text,str) or not text.strip() or len(text)>100000:raise ValueError('text must be nonempty and bounded')
    # Explicit user-supplied source; verify already-imported modules do not shadow it.
    sys.path.insert(0,str(publisher_root.resolve()))
    import prompt_contract,tts_tasks
    for module in (prompt_contract,tts_tasks):
        if Path(module.__file__).resolve().parent!=publisher_root.resolve():raise ValueError('publisher prompt module identity mismatch')
    if digest(checkpoint/'tokenizer.json')!=PINNED_TOKENIZER_SHA256:raise ValueError('unsupported tokenizer identity')
    from tokenizers import Tokenizer
    class TokenizerAdapter:
        def __init__(self):self.tokenizer=Tokenizer.from_file(str(checkpoint/'tokenizer.json'))
        def encode(self,text,add_special_tokens=False):return self.tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
        def convert_tokens_to_ids(self,text):return self.tokenizer.token_to_id(text)
    request=tts_tasks.build_task_generation_kwargs('IP',text=text,language_id='en',controls={'ip':'UK_George'})
    ids,positions,latent=prompt_contract.build_input_ids(TokenizerAdapter(),'dense',request['prompt'],request['text'],instruction=request['instruction'])
    if positions or latent!=-1:raise ValueError('this runner supports built-in voice without reference audio')
    return ids

class NativeMingSession:
    """Request-serialized resident arenas. Cancellation never publishes partial audio."""
    def __init__(self,checkpoint,build_root,*,capacity=256,max_frames=100,cached_backbone=False):
        self.descriptor=load_profile(cached=cached_backbone)
        shell=native_shell_spec(self.descriptor)
        minimum=self.descriptor['minimum_frames']
        if type(max_frames) is not int or not minimum<=max_frames<=512:raise ValueError('invalid frame budget')
        import uuid
        self.instance_id=uuid.uuid4().hex
        self.model_identity={'weights_sha256':digest(checkpoint/'model.safetensors'),'tokenizer_sha256':digest(checkpoint/'tokenizer.json')}
        if self.model_identity!={'weights_sha256':PINNED_WEIGHTS_SHA256,'tokenizer_sha256':PINNED_TOKENIZER_SHA256}:
            raise ValueError('native Ming runner requires the pinned checkpoint and tokenizer')
        self.capacity=capacity;self.max_frames=max_frames;self.lock=threading.Lock();self.cached_backbone=cached_backbone
        if cached_backbone:
            from version.v8.tts.build_ming3_cached_backbone_circuit import build_circuit as selected_backbone
        else:selected_backbone=backbone
        self.graphs={'backbone':selected_backbone(capacity=capacity),'aggregator':aggregator(),'dit':dit(),
                     'decoder':decoder(capacity=max_frames)}
        latent=self.graphs['dit']['activation_buffers']['latent']['shape'][-1]
        steps=len(self.descriptor['sampling']['times'])-1
        self.graphs['update']=update(latent,steps)
        self.vocabulary=self.graphs['backbone']['contract']['runtime_invariants']['vocab']
        self.bundles={};self.exports={};self.identities={}
        # Torch is used only to convert stored BF16 weights to FP32, never inference.
        from safetensors import safe_open
        for name,g in self.graphs.items():
            folder=build_root/name;folder.mkdir(parents=True,exist_ok=False)
            weights={}
            with safe_open(str(checkpoint/'model.safetensors'),framework='pt') as source:
                for key in weight_shapes(g):
                    if name=='dit' and key in constant_weights():weights[key]=constant_weights()[key]
                    elif name=='aggregator' and key=='rotary.positions':weights[key]=np.arange(g['contract']['runtime_invariants']['patches']+1,dtype='f4')
                    else:
                        if name=='backbone':stored='model.'+key if key.startswith('model.') else key
                        else:stored={'aggregator':'linear_proj_audio.','dit':'flowloss.cfm.model.','decoder':'audio.decoder.'}[name]+key
                        weights[key]=source.get_tensor(stored).float().numpy().copy()
            policy={'openmp':True,'c_standard':'gnu11','provider_warnings':'report'}
            if name==self.descriptor['components']['flow_update']['artifact']:
                policy.update(runtime_sources=shell['sources'],extra_exports=[shell['entry']])
            self.bundles[name]=compile_block(folder,g,weights,**policy);del weights
            self.exports[name]=load_exports(folder,self.bundles[name][3])
            self.identities[name]={'library_sha256':digest(folder/'generated.so'),
                'build_identity_sha256':json.loads((folder/'build-identity.json').read_text())['build_identity_sha256'],
                'build_manifest_sha256':digest(folder/'build-identity.json'),'arena_bytes':len(self.bundles[name][1])}
        self.plan=resolve_latent_execution(self.descriptor,self.exports)
        self.model_identity['execution_descriptor_sha256']=self.plan['descriptor_sha256']
        (build_root/'execution-descriptor.json').write_text(json.dumps(self.descriptor,indent=2)+'\n')
        self.driver=getattr(self.bundles[self.plan['components']['flow_update']['artifact']][3],self.plan['shell_entry'])
        self.driver.argtypes=[C.POINTER(LatentLoop)];self.driver.restype=C.c_int

    def synthesize(self,token_ids,*,seed=731,should_cancel=None):
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        if not isinstance(token_ids,(list,tuple)) or not token_ids or any(type(i) is not int or not 0<=i<self.vocabulary for i in token_ids):
            raise ValueError('invalid token IDs')
        if len(token_ids)+self.plan['minimum_frames']-1>self.plan['capacity']:raise ValueError('prompt leaves insufficient minimum generation capacity')
        with self.lock:return self._synthesize(token_ids,seed,should_cancel)

    def _synthesize(self,ids,seed,should_cancel):
        errors=[]
        def check_cancel(context=None):
            try:return int(bool(should_cancel and should_cancel()))
            except BaseException as exc:errors.append(exc);return 1
        callback=CANCEL(check_cancel)
        request,latents,valid,keepalive=bind_latent_request(self.plan,self.bundles,ids,seed,callback)
        start=time.monotonic();status=self.driver(C.byref(request));generation_seconds=time.monotonic()-start
        if errors:raise errors[0]
        if status==2:raise IncompleteSynthesis('context/frame budget exhausted before stop; no waveform published')
        if status==-7:raise CancelledSynthesis('native generation cancelled')
        if status:raise RuntimeError(f'native generation returned {status}')
        if check_cancel():
            if errors:raise errors[0]
            raise CancelledSynthesis('cancelled before audio decoding')
        decoder_plan=self.plan['components']['decoder'];db=self.bundles[decoder_plan['artifact']];frames=valid.value
        def view(name):
            port=decoder_plan['ports'][name]
            dtype=np.int32 if port['dtype']=='i32' else np.float32
            return np.ndarray(tuple(port['shape']),dtype,buffer=db[1],offset=port['offset'])
        view('latent')[:frames]=latents[:frames];view('request_frames')[:]=frames
        view('window')[:]=np.hanning(self.plan['window_elements']+1)[:-1].astype('f4')
        samples=C.c_int32(-1)
        decode=getattr(db[3],decoder_plan['entries']['run'])
        decode.argtypes=[C.POINTER(C.c_uint8),C.c_size_t,C.POINTER(C.c_int32)];decode.restype=C.c_int
        start=time.monotonic();status=decode(db[1],len(db[1]),C.byref(samples));decoder_seconds=time.monotonic()-start
        if check_cancel():
            if errors:raise errors[0]
            raise CancelledSynthesis('cancelled; completed native decoder output discarded')
        if status:raise RuntimeError(f'native decoder returned {status}')
        if not 0<samples.value<=decoder_plan['ports']['waveform']['elements']:raise RuntimeError('invalid native sample count')
        audio=view('waveform')[:samples.value].copy()
        if not np.isfinite(audio).all():raise RuntimeError('nonfinite waveform')
        return audio,{'frames':frames,'samples':len(audio),'sample_rate':self.plan['sample_rate'],'prompt_tokens':len(ids),
            'generation_seconds':generation_seconds,'decoder_seconds':decoder_seconds,'stop_reason':'model_stop',
            'waveform_peak':float(abs(audio).max()),'pcm_clipped_samples':int((abs(audio)>1).sum())},latents[:frames].copy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','publisher-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cached-backbone',action='store_true');p.add_argument('--text',required=True);p.add_argument('--capacity',type=int,default=256)
    p.add_argument('--max-frames',type=int,default=100);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args()
    if not 1<=args.threads<=64:p.error('threads must be 1..64')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'INCOMPLETE','scope':'experimental native FP32 text-to-WAV; not publisher BF16/RNG parity or listening approval',
        'execution_policy':'full_prefill_then_checked_kv_append' if args.cached_backbone else 'full_prefix_recompute_no_kv_cache','seed':args.seed,'threads':args.threads}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        os.environ['OMP_NUM_THREADS']=str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
        ids=prepare_tokens(args.text,args.checkpoint,args.publisher_root)
        if len(ids)+load_profile(cached=args.cached_backbone)['minimum_frames']-1>args.capacity:raise ValueError('prompt leaves insufficient generation capacity')
        report['tokens']=ids;report['checkpoint_sha256']=digest(args.checkpoint/'model.safetensors')
        report['tokenizer_sha256']=digest(args.checkpoint/'tokenizer.json');save()
        start=time.monotonic();session=NativeMingSession(args.checkpoint,args.output/'native',capacity=args.capacity,max_frames=args.max_frames,cached_backbone=args.cached_backbone)
        report.update(load_seconds=time.monotonic()-start,artifacts=session.identities);save()
        audio,measurements,latents=session.synthesize(ids,seed=args.seed)
        np.save(args.output/'waveform.npy',audio);np.save(args.output/'latents.npy',latents)
        with wave.open(str(args.output/'speech.wav'),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(measurements['sample_rate'])
            wav.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
        report.update(status='GENERATED_NOT_CERTIFIED',measurements=measurements,wav_sha256=digest(args.output/'speech.wav'))
    except (CancelledSynthesis,KeyboardInterrupt) as exc:report.update(status='CANCELLED',error=str(exc));save();raise
    except IncompleteSynthesis as exc:report.update(status='INCOMPLETE',error=str(exc));save();raise
    except BaseException as exc:report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));save();raise
    save();print(json.dumps(report))
if __name__=='__main__':main()
