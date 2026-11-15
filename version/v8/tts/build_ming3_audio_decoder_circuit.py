#!/usr/bin/env python3
"""Connected offline Ming latents-to-waveform FP32 diagnostic graph."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit as skeleton
from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.scripts.checked_transformer_authoring_v8 import transformer_stack
OUTPUT=ROOT/'version/v8/circuits/ming3_audio_decoder_fp32_bounded.json'

def build_circuit(frames=4,width=896,heads=14,kv_heads=2,depth=24,inner=4864,latent=64,factor=4,window=64,n_fft=3528):
    values=(frames,width,heads,kv_heads,depth,inner,latent,factor,window,n_fft)
    if any(type(v) is not int or v<1 for v in values) or frames>512 or width>2048 or heads>64 or heads%kv_heads or width%heads or (width//heads)%2 or depth>24 or inner>8192 or latent>256 or factor>4 or window>2048 or n_fft<2 or n_fft>8192 or n_fft%2:
        raise ValueError('unsupported bounded audio-decoder geometry')
    t=frames*factor;d=width//heads;kv=kv_heads*d;bins=n_fft//2+1
    graph=skeleton(1,width,heads,1)
    graph['name']=f'ming3_audio_decoder_fp32_bounded_d{depth}'
    graph['native_entry']['function']='ck_ming3_audio_decoder_fp32_bounded'
    graph['contract']['runtime_invariants'].update(tokens=t,mult=1,frames=frames,depth=depth,
        inner=inner,latent=latent,kv_heads=kv_heads,factor=factor,window=window,n_fft=n_fft,
        complete_dit=False,complete_audio_decoder=True,oracle_spectrum_input=False,
        supplied_rotary_tables=False,streaming=False,scope='reference latents through connected FP32 decoder and FP64-envelope waveform; not publisher BF16 parity')
    shapes={'latent':[frames,latent]};ops=[]
    def put(name,kind,kernel,inputs,outputs,params,weights=None):
        op={'id':name,'op':kind,'kernel':kernel,'returns_status':True,
            'graph_slots':{'inputs':inputs,'outputs':outputs},'params':params}
        if weights:op['weight_refs']=weights
        ops.append(op)
    def project(name,source,k,n,rows,prefix,bias=True):
        op=linear(name,source,name,k,n,rows,prefix)
        if not bias:
            op['op']='linear_rows_checked_no_bias';op['kernel']='linear_rows_checked_no_bias_f32'
            del op['weight_refs']['bias'];del op['params']['call_constants']['linear_bias_elements']
        ops.append(op);shapes[name]=[rows,n]
    project('projection','external:latent',latent,width,frames,'fc1')
    shapes['upsampled']=[t,width]
    put('upsampled','audio_upsample_linear_checked','audio_upsample_linear_frames_f32_checked',
        {'input':'projection'},{'output':'upsampled'},{'T':frames,'C':width,'factor':factor,'call_constants':{
        'linear_input_elements':frames*width,'linear_output_elements':t*width,'linear_frames':frames,'linear_channels':width,'linear_factor':factor}})
    transformer_ops,transformer_shapes=transformer_stack(tokens=t,width=width,heads=heads,kv_heads=kv_heads,
        depth=depth,inner=inner,source='upsampled',weight_namespace='decoder',output='final_norm',window=window)
    ops.extend(transformer_ops);shapes.update(transformer_shapes)
    project('spectral_projection','final_norm',width,2*bins,t,'head.out')
    shapes['real']=[t,bins];shapes['imag']=[t,bins];shapes['window']=[n_fft]
    put('spectrum','audio_spectrum_cartesian_checked','audio_spectrum_cartesian_f32_checked',
        {'input':'spectral_projection'},{'real':'real','imag':'imag'},
        {'F':t,'B':bins,'magnitude_limit':100.,'call_constants':{'spectrum_input_elements':t*2*bins,
         'spectrum_real_elements':t*bins,'spectrum_imag_elements':t*bins,'spectrum_frames':t,'spectrum_bins':bins}})
    hop=n_fft//4
    if hop<1:raise ValueError('FFT must support positive quarter-hop')
    crop=(n_fft-hop)//2;padded=(t-1)*hop+n_fft;samples=padded-2*crop
    shapes['output']=[samples]
    put('waveform','audio_istft_complex_window_checked','audio_istft_complex_window_f32_checked',
        {'real':'real','imag':'imag','window':'external:window'},{'output':'output'},
        {'F':t,'N_fft':n_fft,'O':samples,'Q':2*padded+n_fft,'call_constants':{
         'istft_spectral_elements':t*bins,'istft_window_elements':n_fft,'istft_frames':t,
         'istft_n_fft':n_fft,'istft_hop':hop,'istft_left_crop':crop,'istft_right_crop':crop,
         'istft_output_elements':samples,'istft_scratch_elements':2*padded+n_fft}})
    graph['contract']['runtime_invariants'].update(hop=hop,samples=samples)
    graph['activation_buffers']={k:{'shape':v} for k,v in shapes.items()}
    graph['activation_bindings']={k:k for k in shapes}
    graph['block_types']['block']['body']['ops']=ops
    graph['semantic_checkpoints']['exports']={op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
        'checkpoints':[{'id':'ming3.decoder.'+op['id']+'.'+port,'producer':op['id'],'tensor':tensor,
            'logical_layout':'contiguous','axis_names':['row','channel'],'storage_dtype':'fp32'}
            for port,tensor in op['graph_slots']['outputs'].items()]} for op in ops}
    return graph

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    content=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if OUTPUT.read_text()!=content:raise SystemExit('audio decoder circuit stale')
    else:OUTPUT.write_text(content)
if __name__=='__main__':main()
