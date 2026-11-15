#!/usr/bin/env python3
"""Generated offline waveform reconstruction from supplied Cartesian spectra."""
from __future__ import annotations
import argparse,json
from pathlib import Path
try:
    from version.v8.tts.build_ming3_flow_step_circuit import SOURCE_REVISION,CHECKPOINT_REVISION
except ModuleNotFoundError:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from version.v8.tts.build_ming3_flow_step_circuit import SOURCE_REVISION,CHECKPOINT_REVISION
ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'version/v8/circuits/ming3_tts_waveform_from_spectrum_bounded.json'

def build_circuit(frames=4,n_fft=3528,hop=882):
    if any(type(x) is not int for x in (frames,n_fft,hop)) or not 1<=frames<=512 or not 2<=n_fft<=8192 or n_fft%2 or not 1<=hop<=n_fft//2:
        raise ValueError('invalid bounded spectrum geometry')
    crop=(n_fft-hop)//2; padded=(frames-1)*hop+n_fft
    count=frames*(n_fft//2+1);samples=padded-2*crop;scratch=2*padded+n_fft
    buffers={'real':{'shape':[frames,n_fft//2+1]},'imag':{'shape':[frames,n_fft//2+1]},
             'window':{'shape':[n_fft]},'output':{'shape':[samples]}}
    op={'id':'inverse_transform','op':'audio_istft_complex_window_checked',
        'kernel':'audio_istft_complex_window_bf16_envelope_f32_checked','returns_status':True,
        'graph_slots':{'inputs':{name:'external:'+name for name in ('real','imag','window')},'outputs':{'output':'output'}},
        'params':{'F':frames,'N_fft':n_fft,'O':samples,'Q':scratch,'call_constants':{
            'istft_spectral_elements':count,'istft_window_elements':n_fft,'istft_frames':frames,
            'istft_n_fft':n_fft,'istft_hop':hop,'istft_left_crop':crop,'istft_right_crop':crop,
            'istft_output_elements':samples,'istft_scratch_elements':scratch}}}
    return {'version':3,'name':'ming3_tts_waveform_from_spectrum_bounded','family':'bounded_graph',
        'checked_native_entry':True,'contract':{'runtime_invariants':{'inference_only':True,
            'complete_synthesis':False,'oracle_spectrum_input':True,'streaming':False,
            'production_kernel_heap_allocation':False,'source_revision':SOURCE_REVISION,
            'checkpoint_revision':CHECKPOINT_REVISION,'frames':frames,'n_fft':n_fft,'hop':hop}},
        'activation_buffers':buffers,'activation_bindings':{key:key for key in buffers},
        'native_entry':{'function':'ck_ming3_waveform_from_spectrum_bounded',
            'params':[{'c_type':'uint8_t *','name':'arena'},{'c_type':'size_t','name':'arena_bytes'}],
            'arena':{'pointer':'arena','bytes':'arena_bytes'}},
        'sequence':['waveform'],'block_types':{'waveform':{'sequence':['header','body','footer'],
            'header':[],'body':{'type':'dense','ops':[op]},'footer':[]}}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--check',action='store_true');a=p.parse_args()
    content=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if a.output.read_text()!=content:raise SystemExit('waveform circuit stale')
    else:a.output.write_text(content)
if __name__=='__main__':main()
