#!/usr/bin/env python3
"""Experimental native Ming FP32 text-to-WAV; no publisher neural fallback.

Python prepares tokens, maps weights, invokes one native generation loop and one
native decoder, and writes WAV. Full-prefix recomputation is deliberate. This is
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
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from version.v8.tts.native_guided_flow import Model,Update,Request,ENTRY,NOISE,CANCEL,P,U,S,offsets,schedule
from version.v8.tts.native_latent_loop import LatentLoop
from version.v8.tts.capture_ming3_synthesis_reference import SOURCE_REVISION,digest

class IncompleteSynthesis(RuntimeError):pass


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
    def __init__(self,checkpoint,build_root,*,capacity=256,max_frames=100):
        if type(max_frames) is not int or not 5<=max_frames<=512:raise ValueError('max_frames must be 5..512')
        self.capacity=capacity;self.max_frames=max_frames;self.lock=threading.Lock()
        self.graphs={'backbone':backbone(capacity=capacity),'aggregator':aggregator(),'dit':dit(),
                     'update':update(64),'decoder':decoder(capacity=max_frames)}
        self.bundles={};self.offsets={};self.identities={}
        # Torch is used only to convert stored BF16 weights to FP32, never inference.
        from safetensors import safe_open
        for name,g in self.graphs.items():
            folder=build_root/name;folder.mkdir(parents=True,exist_ok=False)
            weights={}
            with safe_open(str(checkpoint/'model.safetensors'),framework='pt') as source:
                for key in weight_shapes(g):
                    if name=='dit' and key in constant_weights():weights[key]=constant_weights()[key]
                    elif name=='aggregator' and key=='rotary.positions':weights[key]=np.arange(2,dtype='f4')
                    else:
                        if name=='backbone':stored='model.'+key if key.startswith('model.') else key
                        else:stored={'aggregator':'linear_proj_audio.','dit':'flowloss.cfm.model.','decoder':'audio.decoder.'}[name]+key
                        weights[key]=source.get_tensor(stored).float().numpy().copy()
            policy={'openmp':True,'c_standard':'gnu11','provider_warnings':'report'}
            if name=='update':policy.update(runtime_sources=['src/runtime/checked_guided_flow.c','src/runtime/checked_latent_loop.c'],
                extra_exports=['ck_run_checked_latent_loop','audio_noise_counter_gaussian_f32'])
            self.bundles[name]=compile_block(folder,g,weights,**policy);del weights
            self.offsets[name]=offsets(folder)
            self.identities[name]={'library_sha256':digest(folder/'generated.so'),
                'build_identity_sha256':digest(folder/'build-identity.json'),'arena_bytes':len(self.bundles[name][1])}
        self.driver=self.bundles['update'][3].ck_run_checked_latent_loop
        self.driver.argtypes=[C.POINTER(LatentLoop)];self.driver.restype=C.c_int

    def synthesize(self,token_ids,*,seed=731,should_cancel=None):
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        if not isinstance(token_ids,(list,tuple)) or not token_ids or any(type(i) is not int or not 0<=i<151936 for i in token_ids):
            raise ValueError('invalid token IDs')
        if len(token_ids)+4>self.capacity:raise ValueError('prompt leaves insufficient minimum generation capacity')
        with self.lock:return self._synthesize(token_ids,seed,should_cancel)

    def _synthesize(self,ids,seed,should_cancel):
        bb=self.bundles['backbone'];ab=self.bundles['aggregator'];mb=self.bundles['dit'];ub=self.bundles['update']
        b=self.offsets['backbone'];a=self.offsets['aggregator'];m=self.offsets['dit'];u=self.offsets['update']
        model=Model(C.cast(mb[0],ENTRY),C.cast(mb[1],U),len(mb[1]),m['latent'],m['time'],m['condition'],m['history'],m['output']+128*4)
        upd=Update(ENTRY(('ck_ming3_sampler_update',ub[3])),C.cast(ub[1],U),len(ub[1]),
            *[u[k] for k in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output')])
        cond=np.zeros(896,'f4');history=np.zeros(64,'f4');times=schedule();workspace=np.zeros(704,'f4');patch=np.zeros(64,'f4')
        errors=[]
        def check_cancel(context=None):
            try:return int(bool(should_cancel and should_cancel()))
            except BaseException as exc:errors.append(exc);return 1
        callback=CANCEL(check_cancel)
        flow=Request(model,upd,NOISE(('audio_noise_counter_gaussian_f32',ub[3])),cond.ctypes.data_as(P),history.ctypes.data_as(P),
            times.ctypes.data_as(P),64,896,64,10,2.,.25,0.,seed,0,workspace.ctypes.data_as(P),704,patch.ctypes.data_as(P),64,callback,None)
        staging=np.zeros((self.max_frames,64),'f4');latents=np.full_like(staging,np.nan);valid=S(0)
        request=LatentLoop(ENTRY(('ck_ming3_backbone_embed',bb[3])),ENTRY(('ck_ming3_backbone_prefix',bb[3])),C.cast(ab[0],ENTRY),
            C.cast(bb[1],U),C.cast(ab[1],U),len(bb[1]),len(ab[1]),*[b[k] for k in ('request_rows','embeddings','final_norm','stop_logits')],
            *[a[k] for k in ('latent','class_id','output')],896,64,self.capacity,len(ids),self.max_frames,5,C.pointer(flow),
            staging.ctypes.data_as(P),latents.ctypes.data_as(P),staging.size,latents.size,C.pointer(valid))
        bb[2]('prompt_ids',np.int32)[:len(ids)]=ids
        start=time.monotonic();status=self.driver(C.byref(request));generation_seconds=time.monotonic()-start
        if errors:raise errors[0]
        if status==2:raise IncompleteSynthesis('context/frame budget exhausted before stop; no waveform published')
        if status:raise RuntimeError(f'native generation returned {status}')
        if check_cancel():raise RuntimeError('cancelled before audio decoding')
        db=self.bundles['decoder'];view=db[2];frames=valid.value
        view('latent')[:frames]=latents[:frames];view('request_frames',np.int32)[:]=frames
        view('window')[:]=np.hanning(3529)[:-1].astype('f4')
        start=time.monotonic();status=db[0](db[1],len(db[1]));decoder_seconds=time.monotonic()-start
        if check_cancel():raise RuntimeError('cancelled; completed native decoder output discarded')
        if status:raise RuntimeError(f'native decoder returned {status}')
        audio=view('output')[:frames*4*882].copy()
        if not np.isfinite(audio).all():raise RuntimeError('nonfinite waveform')
        return audio,{'frames':frames,'samples':len(audio),'sample_rate':44100,'prompt_tokens':len(ids),
            'generation_seconds':generation_seconds,'decoder_seconds':decoder_seconds,'stop_reason':'model_stop',
            'waveform_peak':float(abs(audio).max()),'pcm_clipped_samples':int((abs(audio)>1).sum())},latents[:frames].copy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','publisher-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--text',required=True);p.add_argument('--capacity',type=int,default=256)
    p.add_argument('--max-frames',type=int,default=100);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args()
    if not 1<=args.threads<=64:p.error('threads must be 1..64')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'INCOMPLETE','scope':'experimental native FP32 text-to-WAV; not publisher BF16/RNG parity or listening approval',
        'execution_policy':'full_prefix_recompute_no_kv_cache','seed':args.seed,'threads':args.threads}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        os.environ['OMP_NUM_THREADS']=str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
        ids=prepare_tokens(args.text,args.checkpoint,args.publisher_root)
        report['tokens']=ids;report['checkpoint_sha256']=digest(args.checkpoint/'model.safetensors')
        report['tokenizer_sha256']=digest(args.checkpoint/'tokenizer.json');save()
        start=time.monotonic();session=NativeMingSession(args.checkpoint,args.output/'native',capacity=args.capacity,max_frames=args.max_frames)
        report.update(load_seconds=time.monotonic()-start,artifacts=session.identities);save()
        audio,measurements,latents=session.synthesize(ids,seed=args.seed)
        np.save(args.output/'waveform.npy',audio);np.save(args.output/'latents.npy',latents)
        with wave.open(str(args.output/'speech.wav'),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(44100)
            wav.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
        report.update(status='GENERATED_NOT_CERTIFIED',measurements=measurements,wav_sha256=digest(args.output/'speech.wav'))
    except IncompleteSynthesis as exc:report.update(status='INCOMPLETE',error=str(exc));save();raise
    except BaseException as exc:report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));save();raise
    save();print(json.dumps(report))
if __name__=='__main__':main()
