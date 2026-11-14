"""Complete bounded Clef head, from supplied backbone features to raw option logits.

The generated entry owns all neural execution. This is not backbone execution.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from build_clef_flash_option_summary_circuit import build_circuit as build_summary

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/clef_flash_complete_head_bounded.json'


def build_circuit(*, tokens=6, questions=2, options=3, channels=4096,
                  width=1024, vocabulary=16, heads=None, feedforward=None):
    if heads is None:
        heads = 16 if width == 1024 else 1
    if feedforward is None:
        feedforward = width * 4
    graph = build_summary(tokens=tokens, questions=questions, options=options,
                          channels=channels, width=width, vocabulary=vocabulary,
                          heads=heads, feedforward=feedforward)
    graph['name'] = 'clef_flash_complete_head_bounded'
    graph['native_entry']['function'] = 'ck_clef_complete_head'
    def buffer(name, shape):
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    buffer('question_type_ids', [questions])

    def op(name, kind, kernel, inputs, output, shape, params, weights=None):
        buffer(output, shape)
        result = {'id': name, 'op': kind, 'kernel': kernel, 'returns_status': True,
                  'graph_slots': {'inputs': inputs, 'outputs': {'output': output}},
                  'params': params}
        if weights: result['weight_refs'] = weights
        return result

    def add(name, left, right, rows, cols):
        n = rows * cols
        return op(name, 'add_rows_checked', 'add_rows_checked_f32',
                  {'left': left, 'right': right}, name, [rows, cols],
                  {'N': n, 'call_constants': {'left_elements': n, 'right_elements': n,
                                             'output_elements': n, 'elements': n}})

    def broadcast(name, left, right, rows, cols):
        n = rows * cols
        return op(name, 'broadcast_add_rows_checked', 'broadcast_add_rows_checked_f32',
                  {'left': left, 'right': right}, name, [rows, cols],
                  {'R': rows, 'C': cols, 'X': n, 'call_constants': {
                      'left_elements': n, 'right_elements': cols, 'output_elements': n,
                      'scratch_elements': n, 'rows': rows, 'channels': cols}})

    def norm(name, source, rows, cols, stem):
        return op(name, 'layernorm', 'layernorm_rows_checked_param_epsilon_f32',
                  {'input': source}, name, [rows, cols],
                  {'M': rows, 'C': cols, 'Q': rows*(cols+2), 'epsilon': 1e-5,
                   'call_constants': {'norm_input_elements': rows*cols, 'norm_input_stride': cols,
                      'norm_gamma_elements': cols, 'norm_beta_elements': cols,
                      'norm_output_elements': rows*cols, 'norm_output_stride': cols,
                      'norm_scratch_elements': rows*(cols+2), 'norm_rows': rows, 'norm_channels': cols}},
                  {'gamma': stem+'.weight', 'beta': stem+'.bias'})

    def linear(name, source, rows, inp, out, stem):
        return op(name, 'linear_rows_checked', 'linear_rows_checked_f32',
                  {'input': source}, name, [rows, out],
                  {'M': rows, 'K': inp, 'N': out, 'call_constants': {
                      'linear_input_elements': rows*inp, 'linear_input_stride': inp,
                      'linear_weight_elements': out*inp, 'linear_weight_stride': inp,
                      'linear_bias_elements': out, 'linear_output_elements': rows*out,
                      'linear_output_stride': out, 'linear_rows': rows,
                      'linear_input_channels': inp, 'linear_output_channels': out}},
                  {'weight': stem+'.weight', 'bias': stem+'.bias'})

    def gelu(name, source, rows, cols):
        return op(name, 'gelu', 'gelu_rows_erf_checked_f32', {'input': source}, name,
                  [rows, cols], {'M': rows, 'C': cols, 'Q': rows*cols, 'call_constants': {
                      'gelu_input_elements': rows*cols, 'gelu_input_stride': cols,
                      'gelu_output_elements': rows*cols, 'gelu_output_stride': cols,
                      'gelu_scratch_elements': rows*cols, 'gelu_rows': rows, 'gelu_channels': cols}})

    def attention(ops, name, query, memory, memory_rows, stem):
        for letter, source, rows in [('q',query,questions),('k',memory,memory_rows),('v',memory,memory_rows)]:
            ops.append(linear(name+'_'+letter,source,rows,width,width,stem+'.'+letter+'_proj'))
        ops.append(op(name+'_context','attention_cross_token_major_checked',
            'attention_cross_token_major_checked_f32',
            {'query':name+'_q','key':name+'_k','value':name+'_v'},name+'_context',[questions,width],
            {'Q':questions,'S':memory_rows,'H':heads,'D':width//heads,'X':memory_rows+questions*width,
             'call_constants':{'query_elements':questions*width,'key_elements':memory_rows*width,
                'value_elements':memory_rows*width,'output_elements':questions*width,
                'scratch_elements':memory_rows+questions*width,'query_tokens':questions,
                'memory_tokens':memory_rows,'heads':heads,'head_dim':width//heads}}))
        ops.append(linear(name+'_out',name+'_context',questions,width,width,stem+'.out_proj'))
        return name+'_out'

    def block(name, ops):
        graph['sequence'].append(name)
        graph['block_types'][name]={'sequence':['header','body','footer'],'header':[],
            'body':{'type':'dense','ops':ops},'footer':[]}

    fields=[op('type_embedding','embedding_lookup_token_major_checked','audio_text_embedding_token_major_f32',
        {'ids':'external:question_type_ids'},'type_embedding',[questions,width],
        {'T':questions,'V':3,'C':width,'call_constants':{'id_elements':questions,'table_elements':3*width,
            'vocabulary':3,'channels':width,'table_stride':width,'output_elements':questions*width,
            'tokens':questions,'output_stride':width}}, {'table':'type_embedding.weight'}),
        add('fields_summary','question_projected','option_summary_normalized',questions,width),
        broadcast('fields_global','fields_summary','global_projected',questions,width),
        add('fields_input','fields_global','type_embedding',questions,width)]
    block('field_assembly',fields)
    previous='fields_input'
    for i in range(4):
        name=f'fields{i}'; stem=f'layers.{i}'; ops=[]
        ops.append(norm(name+'_norm1',previous,questions,width,stem+'.norm1'))
        attn=attention(ops,name+'_self',name+'_norm1',name+'_norm1',questions,stem+'.self_attn')
        ops.append(add(name+'_self_residual',previous,attn,questions,width))
        ops.append(norm(name+'_norm2',name+'_self_residual',questions,width,stem+'.norm2'))
        attn=attention(ops,name+'_cross',name+'_norm2','memory',tokens,stem+'.multihead_attn')
        ops.append(add(name+'_cross_residual',name+'_self_residual',attn,questions,width))
        ops.append(norm(name+'_norm3',name+'_cross_residual',questions,width,stem+'.norm3'))
        ops.append(linear(name+'_ff1',name+'_norm3',questions,width,feedforward,stem+'.linear1'))
        ops.append(gelu(name+'_gelu',name+'_ff1',questions,feedforward))
        ops.append(linear(name+'_ff2',name+'_gelu',questions,feedforward,width,stem+'.linear2'))
        previous=name+'_output'
        ops.append(add(previous,name+'_cross_residual',name+'_ff2',questions,width))
        block(name,ops)

    def pair(name, left, right, cols, features=False, vjp="cosine_similarity"):
        kind='indexed_pair_features_checked' if features else 'indexed_cosine_rows_checked'
        n=options*4*cols if features else options
        params={'G':questions,'R':options,'C':cols,'X':n,'call_constants':{
            'left_elements':questions*cols,'right_elements':options*cols,'indices_elements':options,
            'output_elements':n,'scratch_elements':n,'groups':questions,'rows':options,'channels':cols}}
        if not features: params['epsilon']=1e-12 if vjp=='normalized_dot' else 1e-8
        result = op(name,kind,kind+'_f32',{'left':left,'right':right,'indices':'option_question_indices'},
                  name,[options,4*cols] if features else [options],params)
        if not features: result['training'] = {'vjp': vjp}
        return result

    def scalar(name, weight, sigmoid=False):
        kind='scalar_sigmoid_checked' if sigmoid else 'scalar_exp_clamped_checked'
        params={'call_constants':{'input_elements':1,'output_elements':1}}
        if not sigmoid:params['maximum']=math.log(100.)
        return op(name,kind,kind+'_f32',{},name,[1],params,{'input':weight})

    def scale(name, source, gain):
        return op(name,'scale_vector_checked','scale_vector_checked_f32',{'input':source,'gain':gain},
            name,[options],{'R':options,'X':options,'call_constants':{'input_elements':options,
                'gain_elements':1,'output_elements':options,'scratch_elements':options,'rows':options}})

    scoring=[norm('final_fields',previous,questions,width,'field_norm'),
        norm('final_options','evidence1_output',options,width,'option_norm'),
        broadcast('prior_anchor','question_vectors','global_vector',questions,channels),
        pair('prior_cosine','prior_anchor','lexical_option_vectors',channels,vjp='normalized_dot'),
        pair('joint_cosine','final_fields','final_options',width),
        pair('scorer_features','final_fields','final_options',width,True),
        linear('scorer_hidden','scorer_features',options,4*width,width,'residual_scorer.0'),
        gelu('scorer_gelu','scorer_hidden',options,width),
        linear('scorer_residual','scorer_gelu',options,width,1,'residual_scorer.3'),
        scalar('prior_scale','prior_logit_scale'),scalar('joint_scale','joint_logit_scale'),
        scalar('residual_mix','residual_gate',True),
        scale('prior_scores','prior_cosine','prior_scale'),
        scale('scaled_joint_cosine','joint_cosine','joint_scale'),
        add('joint_scores','scaled_joint_cosine','scorer_residual',options,1),
        scale('gated_joint_scores','joint_scores','residual_mix'),
        add('option_logits','prior_scores','gated_joint_scores',options,1)]
    block('decision_scoring',scoring)
    return graph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    text=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if OUTPUT.read_text()!=text:raise SystemExit('complete head circuit differs from authoring')
    else:OUTPUT.write_text(text)

if __name__=='__main__':main()
