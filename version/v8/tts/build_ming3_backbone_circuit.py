"""Checked dynamic-prefix Ming backbone; explicit full-prefix recomputation baseline."""
import copy
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.scripts.checked_transformer_authoring_v8 import transformer_stack
from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.scripts.checked_circuit_authoring_v8 import runtime_extent as _extent, bind_runtime_scalars as _bind

def build_circuit(capacity=128,width=896,heads=14,kv_heads=2,depth=24,inner=4864,vocab=151936):
    if type(capacity) is not int or not 1<=capacity<=512 or type(vocab) is not int or not 1<=vocab<=200000:
        raise ValueError('unsupported backbone capacity/vocabulary')
    ops,shapes=transformer_stack(tokens=capacity,width=width,heads=heads,kv_heads=kv_heads,depth=depth,
        inner=inner,source='external:embeddings',weight_namespace='model',output='final_norm',window=capacity)
    ops.append(linear('stop_logits','final_norm','stop_logits',width,2,capacity,'stop_head'))
    g={'version':3,'name':f'ming3_backbone_fp32_bounded_d{depth}','family':'bounded_graph',
        'checked_native_entry':True,'contract':{'runtime_invariants':{
            'inference_only':True,'complete_synthesis':False,'publisher_bf16_parity':False,
            'arithmetic':'FP32 storage with map-declared reductions','tokens':capacity,'width':width,
            'heads':heads,'kv_heads':kv_heads,'depth':depth,'inner':inner,'mult':1,'capacity':capacity,'vocab':vocab,
            'execution_policy':'full_prefix_recompute_no_kv_cache',
            'scope':'FP32 embeddings/backbone/stop logits with checked runtime prefix'}},
        'native_entry':{'function':'ck_ming3_backbone_fp32',
            'params':[{'c_type':'uint8_t *','name':'arena'},{'c_type':'size_t','name':'arena_bytes'}],
            'arena':{'pointer':'arena','bytes':'arena_bytes'},
            'segments':[{'function':'ck_ming3_backbone_embed','last_op':'embeddings'},
                        {'function':'ck_ming3_backbone_prefix','last_op':'stop_logits'}]},
        'block_types':{'block':{'sequence':['header','body','footer'],'header':[],
                              'body':{'type':'dense','ops':ops},'footer':[]}},
        'activation_buffers':{k:{'shape':v} for k,v in shapes.items()},
        'semantic_checkpoints':{'schema':'cke.semantic_checkpoint_contract','schema_version':1,'exports':{}}}
    g['runtime_constants']={'value_elements':1,'phoneme_count':1,'max_duration':capacity,'expanded_capacity':capacity}
    limits={'embed_rows':capacity,'prefix_rows':capacity,'mlp_elements':capacity*inner}
    g['runtime_lengths']={k:{'producer':'check_'+k,'result':'valid_extent','capacity':v,'allow_zero':False} for k,v in limits.items()}
    headers=[_extent('check_prefix_rows','runtime_extent_sum','request_rows','prefix_rows','prefix_rows',capacity=capacity),
        _extent('check_mlp_elements','runtime_extent_scale','prefix_rows','mlp_elements','mlp_elements',factor=inner,capacity=capacity*inner)]
    for op in ops:
        kind=op['op']
        if kind.startswith('linear_rows'):bindings={'linear_rows':'prefix_rows'}
        elif kind=='rmsnorm_rows_checked':bindings={'rows':'prefix_rows'}
        elif kind in ('token_head_layout_checked','rope_qk_bounded','attention_sliding_bounded'):bindings={'tokens':'prefix_rows'}
        elif kind=='audio_scaled_sum_strided_checked':bindings={'sum_rows':'prefix_rows'}
        elif kind=='silu_mul_rows_checked':bindings={'elements':'mlp_elements'}
        else:raise ValueError('unbound prefix consumer '+kind)
        _bind(op,**bindings)
    embed={'id':'embeddings','op':'embedding_lookup_token_major_checked','kernel':'audio_text_embedding_token_major_f32',
        'returns_status':True,'weight_refs':{'table':'model.embed_tokens.weight'},
        'graph_slots':{'inputs':{'ids':'external:prompt_ids'},'outputs':{'output':'external:embeddings'}},
        'params':{'T':capacity,'V':vocab,'C':width,'call_constants':{'id_elements':capacity,'table_elements':vocab*width,
            'vocabulary':vocab,'channels':width,'table_stride':width,'output_elements':capacity*width,'tokens':capacity,'output_stride':width}}}
    _bind(embed,tokens='embed_rows')
    g['sequence']=['embedding','block']
    g['block_types']['embedding']={'sequence':['header','body','footer'],
        'header':[_extent('check_embed_rows','runtime_extent_sum','request_rows','embed_rows','embed_rows',capacity=capacity)],
        'body':{'type':'dense','ops':[embed]},'footer':[]}
    g['block_types']['block']['header']=headers;g['block_types']['block']['body']['ops']=ops
    used={'request_rows','embed_rows','prefix_rows','mlp_elements','prompt_ids','embeddings'}
    for op in ops:
        for ports in op['graph_slots'].values():used.update(v.removeprefix('external:') for v in ports.values())
    shapes={k:v for k,v in g['activation_buffers'].items() if k in used}
    shapes.update({k:{'shape':[1]} for k in ('request_rows','embed_rows','prefix_rows','mlp_elements')})
    shapes.update(prompt_ids={'shape':[capacity]},embeddings={'shape':[capacity,width],'lifetime':'sequence'},stop_logits={'shape':[capacity,2]})
    g['activation_buffers']=shapes;g['activation_bindings']={k:k for k in shapes}
    g['semantic_checkpoints']['exports']={op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
        'checkpoints':[{'id':'ming3.backbone.'+op['id']+'.'+port,'producer':op['id'],'tensor':tensor.removeprefix('external:'),
          'logical_layout':'contiguous','axis_names':['row','channel'],'storage_dtype':'fp32'} for port,tensor in op['graph_slots']['outputs'].items()]} for op in [embed,*ops]}
    g['execution_exports']={'schema':'cke.checked.execution_exports.v1',
        'entries':{'embed':'ck_ming3_backbone_embed','prefill':'ck_ming3_backbone_prefix'},
        'ports':{name:execution_port(g,buffer) for name,buffer,dtype in (
            ('request_rows','request_rows','i32'),('token_ids','prompt_ids','i32'),
            ('embeddings','embeddings','fp32'),('hidden','final_norm','fp32'),('stop_logits','stop_logits','fp32'))}}
    return g
