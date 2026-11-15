#!/usr/bin/env python3
"""Ming DiT block in explicit FP32 arithmetic with supplied rotary tables.

This establishes graph composition against an FP32 reference. It does not claim
publisher BF16 storage equivalence, a full DiT, or native speech generation.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.scripts.checked_circuit_authoring_v8 import linear
OUTPUT=ROOT/'version/v8/circuits/ming3_tts_dit_block_fp32_bounded.json'

def build_circuit(tokens=3,width=1024,heads=16,mult=4):
    if any(type(v) is not int for v in (tokens,width,heads,mult)) or not 1<=tokens<=128 or not 1<=heads<=64 or not 2<=width<=2048 or width%heads or (width//heads)%2 or not 1<=mult<=8:
        raise ValueError('unsupported bounded block geometry')
    d=width//heads;inner=width*mult;n=tokens*width
    shapes={key:[tokens,width] for key in ('input','norm1','q','k','v','q_rot','k_rot','attention','attention_projection','residual1','norm2','down','output')}
    shapes.update(cosine=[tokens,d//2],sine=[tokens,d//2],up=[tokens,inner],activated=[tokens,inner])
    def op(name,kind,kernel,inputs,output,params,weights=None):
        o={'id':name,'op':kind,'kernel':kernel,'returns_status':True,
            'graph_slots':{'inputs':inputs,'outputs':{'output':output}},'params':params}
        if weights:o['weight_refs']=weights
        return o
    def norm(name,source):
        return op(name,'rmsnorm_rows_checked','rmsnorm_rows_checked_f32',{'input':source},name,
            {'R':tokens,'C':width,'eps':1e-6,'call_constants':{'input_elements':n,'input_stride':width,
                'gamma_elements':width,'output_elements':n,'output_stride':width,'rows':tokens,'width':width}},
            {'gamma':name+'.weight'})
    def rotate(name,source):
        return op(name,'rope_pairwise_rows_checked','rope_pairwise_rows_f32_checked',
            {'input':source,'cosine':'external:cosine','sine':'external:sine'},name,
            {'T':tokens,'H':heads,'D':d,'call_constants':{'rotation_input_elements':n,
                'rotation_output_elements':n,'rotation_table_elements':tokens*d//2,
                'rotation_tokens':tokens,'rotation_heads':heads,'rotation_head_dim':d}})
    def add(name,left,right):
        return op(name,'audio_scaled_sum_strided_checked','audio_scaled_sum_strided_f32_checked',
            {'left':left,'right':right},name,{'R':tokens,'C':width,'scale':1.,'call_constants':{
                'sum_left_elements':n,'sum_right_elements':n,'sum_output_elements':n,
                'sum_left_stride':width,'sum_right_stride':width,'sum_output_stride':width,
                'sum_rows':tokens,'sum_columns':width}})
    ops=[norm('norm1','external:input')]
    for key in ('q','k','v'):ops.append(linear(key,'norm1',key,width,width,tokens,'attn.to_'+key))
    ops += [rotate('q_rot','q'),rotate('k_rot','k'),op('attention','attention_full_token_major_checked',
        'attention_full_token_major_f32_checked',{'query':'q_rot','key':'k_rot','value':'v'},'attention',
        {'T':tokens,'H':heads,'D':d,'call_constants':{'attention_query_elements':n,'attention_key_elements':n,
            'attention_value_elements':n,'attention_output_elements':n,'attention_tokens':tokens,
            'attention_heads':heads,'attention_head_dim':d}}),
        linear('attention_projection','attention','attention_projection',width,width,tokens,'attn.to_out.0'),
        add('residual1','external:input','attention_projection'),norm('norm2','residual1'),
        linear('up','norm2','up',width,inner,tokens,'mlp.ff.0.0'),
        op('activated','gelu','gelu_rows_tanh_checked_f32',{'input':'up'},'activated',
            {'M':tokens,'C':inner,'N':inner,'Q':tokens*inner,'call_constants':{'gelu_input_elements':tokens*inner,'gelu_input_stride':inner,
                'gelu_output_elements':tokens*inner,'gelu_output_stride':inner,'gelu_scratch_elements':tokens*inner,
                'gelu_rows':tokens,'gelu_channels':inner}}),
        linear('down','activated','down',inner,width,tokens,'mlp.ff.2'),add('output','residual1','down')]
    return {'version':3,'name':'ming3_tts_dit_block_fp32_bounded','family':'bounded_graph',
        'checked_native_entry':True,'contract':{'runtime_invariants':{'inference_only':True,
            'complete_synthesis':False,'complete_dit':False,'publisher_bf16_parity':False,
            'supplied_rotary_tables':True,'arithmetic':'FP32 storage with map-declared reductions',
            'tokens':tokens,'width':width,'heads':heads,'mult':mult}},
        'activation_buffers':{k:{'shape':v} for k,v in shapes.items()},'activation_bindings':{k:k for k in shapes},
        'native_entry':{'function':'ck_ming3_dit_block_fp32_bounded',
            'params':[{'c_type':'uint8_t *','name':'arena'},{'c_type':'size_t','name':'arena_bytes'}],
            'arena':{'pointer':'arena','bytes':'arena_bytes'}},
        'sequence':['block'],'block_types':{'block':{'sequence':['header','body','footer'],'header':[],
            'body':{'type':'dense','ops':ops},'footer':[]}},
        'semantic_checkpoints':{'schema':'cke.semantic_checkpoint_contract','schema_version':1,
            'exports':{op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
                'checkpoints':[{'id':'ming3.dit.'+op['id'],'producer':op['id'],
                    'tensor':next(iter(op['graph_slots']['outputs'].values())),
                    'logical_layout':'token_major_contiguous','axis_names':['token','channel'],
                    'storage_dtype':'fp32'}]} for op in ops}}}

from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes as weight_shapes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--check',action='store_true');a=p.parse_args();content=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if a.output.read_text()!=content:raise SystemExit('Ming block circuit stale')
    else:a.output.write_text(content)
if __name__=='__main__':main()
