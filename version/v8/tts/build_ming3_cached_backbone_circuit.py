"""Generated Ming prefix/cache seeding and one-row cached continuation."""
import copy
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.tts.build_ming3_backbone_circuit import build_circuit as full
from version.v8.scripts.checked_circuit_authoring_v8 import runtime_extent as _extent, bind_runtime_scalars as _bind

def build_circuit(**kwargs):
    g=full(**kwargs);c=g['contract']['runtime_invariants'];capacity=c['capacity'];h=c['heads'];kh=c['kv_heads'];d=c['width']//h
    tiny=full(**{**kwargs,'capacity':1})
    g['name']+='__cached';c['execution_policy']='full_prefill_then_checked_kv_append'
    shapes=g['activation_buffers'];bindings=g['activation_bindings']
    cache_ops=[]
    for layer in range(c['depth']):
        p=f'layer{layer}_';elements=kh*capacity*d
        for kind in ('k','v'):
            name=p+'cache_'+kind;shapes[name]={'shape':[kh,capacity,d],'lifetime':'sequence'};bindings[name]=name
        op={'id':p+'cache_seed','op':'kv_cache_seed_checked','kernel':'kv_cache_seed_checked_f32','returns_status':True,
            'graph_slots':{'inputs':{'k':p+'k_head','v':p+'v_head','cache_k':'external:'+p+'cache_k','cache_v':'external:'+p+'cache_v'},
                'outputs':{'cache_k':'external:'+p+'cache_k','cache_v':'external:'+p+'cache_v'}},
            'params':{'KV':kh,'T':capacity,'S':capacity,'D':d,'call_constants':{'k_elements':elements,'v_elements':elements,
                'k_cache_elements':elements,'v_cache_elements':elements,'kv_heads':kh,'cache_capacity':capacity,'head_dim':d,'tokens':capacity}}}
        _bind(op,tokens='prefix_rows');cache_ops.append(op)
    g['block_types']['block']['body']['ops'].extend(cache_ops)
    g['native_entry']['segments'][1]['last_op']=cache_ops[-1]['id']
    g['native_entry']['segments'].append({'function':'ck_ming3_backbone_decode','last_op':'decode_stop_logits'})
    g['sequence'].append('decode')
    shapes['decode_position_input']={'shape':[1]};shapes['decode_position_checked']={'shape':[1]}
    bindings['decode_position_input']='decode_position_input';bindings['decode_position_checked']='decode_position_checked'
    g['runtime_lengths']['decode_position']={'producer':'check_decode_position','result':'valid_extent','capacity':capacity-1,'allow_zero':True}
    header=_extent('check_decode_position','runtime_extent_sum','decode_position_input','decode_position','decode_position_checked',capacity=capacity-1)
    ops=copy.deepcopy(tiny['block_types']['block']['body']['ops'])
    for name,b in tiny['activation_buffers'].items():
        if name not in ('request_rows','embed_rows','prefix_rows','mlp_elements','prompt_ids'):
            shapes['decode_'+name]=copy.deepcopy(b);bindings['decode_'+name]='decode_'+name
    for op in ops:
        original=op['id'];op['id']='decode_'+original
        for arg in op.pop('runtime_scalar_bindings',{}):
            op['params']['call_constants'][arg]=c['inner'] if arg=='elements' else 1
        op.pop('consumes_runtime_lengths',None)
        op['graph_slots']={direction:{port:('external:decode_'+value[9:] if value.startswith('external:') else 'decode_'+value)
            for port,value in ports.items()} for direction,ports in op['graph_slots'].items()}
        if op['op']=='rope_qk_bounded':_bind(op,pos_offset='decode_position')
        if op['op']=='attention_sliding_bounded':
            prefix=original.removesuffix('attention');dp='decode_'+prefix;ce=kh*capacity*d
            for name,shape in [(dp+'scores',[h,capacity]),(dp+'staged',[h,d])]:shapes[name]={'shape':shape};bindings[name]=name
            op.update(op='attention_decode_append_checked',kernel='attention_decode_append_checked_f32',
                graph_slots={'inputs':{'q':dp+'q_head','k_new':dp+'k_head','v_new':dp+'v_head',
                    'cache_k':'external:'+prefix+'cache_k','cache_v':'external:'+prefix+'cache_v'},
                    'outputs':{'work_scores':dp+'scores','staged':dp+'staged','out':dp+'attention_head',
                        'cache_k':'external:'+prefix+'cache_k','cache_v':'external:'+prefix+'cache_v'}},
                params={'H':h,'KV':kh,'S':capacity,'D':d,'call_constants':{
                    'q_elements':h*d,'k_new_elements':kh*d,'v_new_elements':kh*d,'k_cache_elements':ce,'v_cache_elements':ce,
                    'score_elements':h*capacity,'staged_output_elements':h*d,'output_elements':h*d,
                    'heads':h,'kv_heads':kh,'cache_capacity':capacity,'head_dim':d,'position':0}})
            _bind(op,position='decode_position')
    g['block_types']['decode']={'sequence':['header','body','footer'],'header':[header],'body':{'type':'dense','ops':ops},'footer':[]}
    # Semantic exports follow actual generated operations, including persistent KV.
    g['semantic_checkpoints']['exports']={op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
        'checkpoints':[{'id':'ming3.cached.'+op['id']+'.'+port,'producer':op['id'],'tensor':value.removeprefix('external:'),
            'logical_layout':'contiguous','axis_names':['element'],'storage_dtype':'fp32'} for port,value in op['graph_slots']['outputs'].items()]}
        for block in g['block_types'].values() for op in block['body']['ops']}
    g['execution_exports']['entries']['decode']='ck_ming3_backbone_decode'
    g['execution_exports']['ports'].update({name:execution_port(g,buffer) for name,buffer,dtype in (
        ('decode_input','decode_embeddings','fp32'),('decode_hidden','decode_final_norm','fp32'),
        ('decode_stop_logits','decode_stop_logits','fp32'),('decode_position','decode_position_input','i32'))})
    return g
