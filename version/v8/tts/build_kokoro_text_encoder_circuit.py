#!/usr/bin/env python3
"""Declare complete pinned acoustic text encoder joined to generated durations.

The circuit, rather than Python at deployment, orders three effective-weight
Conv1D/LayerNorm/LeakyReLU blocks, a zero-state BiLSTM and frame expansion.
"""
import argparse
import json
from pathlib import Path

from build_kokoro_text_embedding_circuit import build_circuit as embedding_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_text_encoder_bounded.json'
TOKENS = 36
CHANNELS = 512
STRIDE = 40
FRAMES = 128


def conv_op(index, source, tokens):
    prefix=f'acoustic_text_encoder.conv{index}'
    output=f'text_conv{index}'
    return {'id':output,'op':'audio_conv1d_checked',
        'kernel':'audio_conv1d_checked_channel_major_f32',
        'returns_status':True,
        'weight_refs':{'weight':f'{prefix}.weight','bias':f'{prefix}.bias'},
        'params':{'I':CHANNELS,'O':CHANNELS,'T':tokens,'K':5,'U':tokens,
                  'Q':CHANNELS*tokens,
                  'call_constants':{
            'conv_input_elements':CHANNELS*STRIDE,'conv_input_stride':STRIDE,
            'conv_weight_elements':CHANNELS*CHANNELS*5,
            'conv_bias_elements':CHANNELS,
            'conv_output_elements':CHANNELS*STRIDE,'conv_output_stride':STRIDE,
            'conv_scratch_elements':CHANNELS*tokens,
            'conv_input_channels':CHANNELS,'conv_output_channels':CHANNELS,
            'conv_input_frames':tokens,'conv_kernel_size':5,
            'conv_stride':1,'conv_padding':2,'conv_output_frames':tokens}},
        'graph_slots':{'inputs':{'input':source},'outputs':{'output':output}}}


def transpose_op(index,source,tokens,channel_to_token=True):
    output=(f'text_conv{index}_token' if channel_to_token else
            f'text_act{index}_channel')
    rows=CHANNELS if channel_to_token else tokens
    columns=tokens if channel_to_token else CHANNELS
    input_stride=STRIDE if channel_to_token else CHANNELS
    output_stride=CHANNELS if channel_to_token else STRIDE
    return {'id':output,'op':'transpose_strided_checked',
        'kernel':'transpose_strided_f32_checked','returns_status':True,
        'params':{'R':rows,'C':columns,'call_constants':{
            'transpose_input_elements':rows*input_stride,
            'transpose_input_stride':input_stride,
            'transpose_output_elements':columns*output_stride,
            'transpose_output_stride':output_stride,
            'transpose_rows':rows,'transpose_columns':columns}},
        'graph_slots':{'inputs':{'input':source},'outputs':{'output':output}}}


def norm_op(index, tokens):
    prefix=f'acoustic_text_encoder.norm{index}'
    output=f'text_norm{index}'
    return {'id':output,'op':'layernorm_rows_param_checked',
        'kernel':'layernorm_rows_checked_param_epsilon_f32',
        'returns_status':True,
        'weight_refs':{'gamma':f'{prefix}.weight','beta':f'{prefix}.bias'},
        'params':{'M':tokens,'C':CHANNELS,'Q':tokens*(CHANNELS+2),
            'epsilon':1e-5,'call_constants':{
            'norm_input_elements':tokens*CHANNELS,
            'norm_input_stride':CHANNELS,
            'norm_gamma_elements':CHANNELS,
            'norm_beta_elements':CHANNELS,
            'norm_output_elements':tokens*CHANNELS,
            'norm_output_stride':CHANNELS,
            'norm_scratch_elements':tokens*(CHANNELS+2),
            'norm_rows':tokens,'norm_channels':CHANNELS}},
        'graph_slots':{'inputs':{'input':f'text_conv{index}_token'},
                       'outputs':{'output':output}}}


def activation_op(index, tokens):
    output=f'text_act{index}'
    return {'id':output,'op':'leaky_relu_strided_checked',
        'kernel':'leaky_relu_strided_f32_checked','returns_status':True,
        'params':{'R':tokens,'C':CHANNELS,'negative_slope':0.2,
                  'call_constants':{
            'leaky_input_elements':tokens*CHANNELS,
            'leaky_input_stride':CHANNELS,
            'leaky_output_elements':tokens*CHANNELS,
            'leaky_output_stride':CHANNELS,
            'leaky_rows':tokens,'leaky_columns':CHANNELS}},
        'graph_slots':{'inputs':{'input':f'text_norm{index}'},
                       'outputs':{'output':output}}}


def lstm_op(tokens):
    prefix='acoustic_text_encoder.lstm'
    return {'id':'text_lstm','op':'audio_lstm_bidirectional_scan',
        'kernel':'audio_lstm_bidirectional_scan_f32','returns_status':True,
        'weight_refs':{kind:f'{prefix}.{kind}' for kind in
                       ('weight_ih','weight_hh','bias_ih','bias_hh')},
        'params':{'T':tokens,'I':CHANNELS,'H':256,
            'input_size':CHANNELS,'hidden_size':256,
            'call_constants':{
            'scan_input_elements':tokens*CHANNELS,
            'scan_weight_ih_elements':2*1024*CHANNELS,
            'scan_weight_hh_elements':2*1024*256,
            'scan_bias_ih_elements':2*1024,'scan_bias_hh_elements':2*1024,
            'scan_output_elements':tokens*CHANNELS,
            'scan_hidden_state_elements':512,
            'scan_cell_state_elements':512,
            'scan_input_stride':CHANNELS,'scan_output_stride':CHANNELS}},
        'graph_slots':{'inputs':{'input':'text_act2'},
                       'outputs':{'output':'text_encoder_features'}}}


def build_circuit(tokens=TOKENS):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= STRIDE:
        raise ValueError('text encoder tokens must be an integer between 2 and 40')
    graph=embedding_circuit(tokens)
    graph['name']='kokoro_text_encoder_bounded'
    graph['native_entry']['function']='ck_kokoro_text_encoder_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_acoustic_text_encoder=True,
        complete_text_encoder=True, complete_waveform=False,
        all_tokens_valid=True, text_encoder_oracle_fed=False)
    ops=[]
    previous='text_features'
    for index in range(3):
        ops.extend((conv_op(index,previous,tokens),
                    transpose_op(index,f'text_conv{index}',tokens),
                    norm_op(index,tokens),activation_op(index,tokens)))
        graph['activation_buffers'][f'text_conv{index}']={'shape':[CHANNELS,STRIDE]}
        graph['activation_buffers'][f'text_conv{index}_token']={'shape':[tokens,CHANNELS]}
        graph['activation_buffers'][f'text_norm{index}']={'shape':[tokens,CHANNELS]}
        graph['activation_buffers'][f'text_act{index}']={'shape':[tokens,CHANNELS]}
        if index<2:
            ops.append(transpose_op(index,f'text_act{index}',tokens,False))
            graph['activation_buffers'][f'text_act{index}_channel']={
                'shape':[CHANNELS,STRIDE]}
            previous=f'text_act{index}_channel'
    ops.append(lstm_op(tokens))
    graph['activation_buffers']['text_encoder_features']={'shape':[tokens,CHANNELS]}
    for name in graph['activation_buffers']:
        graph['activation_bindings'][name]=name
    graph['sequence'].insert(-1,'text_encoder')
    graph['block_types']['text_encoder']={
        'sequence':['header','body','footer'],
        'header':[], 'body':{'type':'dense','ops':ops},'footer':[]}
    graph['required_numerical_contracts']['layernorm_rows_param_checked']={
        'op':'layernorm','template_ops':['layernorm_rows_param_checked'],
        'phases':{'prefill':{
            'contract_id':'layernorm_rows_checked_fp32_ggml_chunked_contracted_param_epsilon',
            'validation':'validated',
            'evidence':'tests/test_v8_kokoro_generated_text_encoder.py'}},
        'checkpoint':{'id':'kokoro.text_encoder.text_norm0',
            'producer':'text_norm0','logical_layout':'token_major',
            'axis_names':['token','channel']}}
    for op_name,provider_op,contract_id,checkpoint,layout,axes in (
        ('audio_conv1d_checked','audio_conv1d_checked',
         'audio_conv1d_checked_scalar_fma_fp32','text_conv0',
         'channel_major',['channel','token']),
        ('transpose_strided_checked','transpose_strided_checked',
         'transpose_strided_f32_checked_exact_fp32','text_conv0_token',
         'token_major',['token','channel']),
        ('leaky_relu_strided_checked','leaky_relu_strided_checked',
         'leaky_relu_strided_f32_checked_exact_fp32','text_act0',
         'token_major',['token','channel']),
        ('audio_duration_expand','audio_duration_expand',
         'audio_duration_expand_token_major_checked_strided_copy_fp32',
         'expand_generated_text_encoder','channel_major',['channel','frame'])):
        graph['required_numerical_contracts'][op_name]={
            'op':provider_op,'template_ops':[op_name],
            'phases':{'prefill':{'contract_id':contract_id,
                'validation':'validated',
                'evidence':'tests/test_v8_kokoro_generated_text_encoder.py'}},
            'checkpoint':{'id':f'kokoro.text_encoder.{checkpoint}',
                'producer':checkpoint,'logical_layout':layout,
                'axis_names':axes}}
    expansion=graph['block_types']['duration_expansion']['body']['ops'][1]
    expansion['id']='expand_generated_text_encoder'
    expansion['kernel']='audio_duration_expand_token_major_f32'
    expansion['graph_slots']['inputs']['features']='text_encoder_features'
    expansion['params']['call_constants']['input_elements']=tokens*CHANNELS
    expansion['params']['call_constants']['input_stride']=CHANNELS
    graph['semantic_checkpoints']['exports'].pop('expand_generated_text_embedding')
    graph['semantic_checkpoints']['exports'][expansion['id']]={
        'section':'body','template_op_id':expansion['id'],'op':expansion['op'],
        'checkpoints':[{'id':'kokoro.text_encoder.expanded','producer':expansion['id'],
            'tensor':'text_expanded','logical_layout':'channel_major',
            'axis_names':['channel','frame'],'storage_dtype':'fp32'}]}
    for op in ops:
        ident=op['id']; tensor=op['graph_slots']['outputs']['output']
        layout='channel_major' if tensor.startswith('text_conv') and not tensor.endswith('_token') or tensor.endswith('_channel') else 'token_major'
        graph['semantic_checkpoints']['exports'][ident]={
            'section':'body','template_op_id':ident,'op':op['op'],
            'checkpoints':[{'id':f'kokoro.text_encoder.{ident}',
                'producer':ident,'tensor':tensor,'logical_layout':layout,
                'axis_names':['channel','token'] if layout=='channel_major' else ['token','channel'],
                'storage_dtype':'fp32'}]}
    return graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    content=json.dumps(build_circuit(),indent=2)+'\n'
    if args.check:
        if args.output.read_text()!=content:
            raise SystemExit('text encoder circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__=='__main__':main()
