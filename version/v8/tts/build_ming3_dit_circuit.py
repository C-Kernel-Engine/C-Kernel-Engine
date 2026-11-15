#!/usr/bin/env python3
"""Full FP32 DiT with native timestep features and rotary tables."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit as block_circuit
from version.v8.scripts.checked_circuit_authoring_v8 import linear
OUTPUT=ROOT/'version/v8/circuits/ming3_tts_dit_fp32_bounded.json'

def build_circuit(width=1024,heads=16,depth=8,latent=64,condition=896,time_width=256,mult=4):
    if any(type(v) is not int or v<1 for v in (depth,latent,condition,time_width)) or depth>8 or latent>256 or condition>4096 or time_width>1024 or time_width<4 or time_width%2:
        raise ValueError('unsupported bounded DiT configuration')
    block=block_circuit(3,width,heads,mult);graph=copy.deepcopy(block)
    graph['name']='ming3_tts_dit_fp32_bounded'+('' if depth==8 else f'_d{depth}');graph['native_entry']['function']='ck_ming3_dit_fp32_bounded'
    inv=graph['contract']['runtime_invariants'];inv.update(complete_dit=True,connected_dit_layers=depth,
        supplied_timestep_features=False,supplied_rotary_tables=False,latent=latent,condition=condition,time_width=time_width,
        scope='all learned DiT layers and native positional features; explicit FP32 execution variant')
    shapes={'latent':[1,latent],'history':[1,latent],'condition':[1,condition],'time_features':[1,time_width],'time':[1],
        'time_fc1':[1,width],'time_active':[1,width],'time_emb':[1,width],'condition_emb':[1,width],
        'conditioning':[1,width],'latent_emb':[1,width],'history_emb':[1,width],
        'pair':[2,width],'joined':[3,width],'cosine':[3,width//heads//2],
        'sine':[3,width//heads//2],'final_norm':[3,width],'output':[3,latent]}
    ops=[linear('time_fc1','time_features','time_fc1',time_width,width,1,'t_embedder.time_mlp.0'),
        {'id':'time_active','op':'silu_strided_checked','kernel':'silu_strided_f32_checked','returns_status':True,
         'graph_slots':{'inputs':{'input':'time_fc1'},'outputs':{'output':'time_active'}},
         'params':{'R':1,'C':width,'call_constants':{'silu_input_elements':width,'silu_input_stride':width,
             'silu_output_elements':width,'silu_output_stride':width,'silu_rows':1,'silu_columns':width}}},
        linear('time_emb','time_active','time_emb',width,width,1,'t_embedder.time_mlp.2'),
        linear('condition_emb','external:condition','condition_emb',condition,width,1,'c_embedder.cond_embedder'),
        linear('history_emb','external:history','history_emb',latent,width,1,'x_embedder'),
        linear('latent_emb','external:latent','latent_emb',latent,width,1,'x_embedder')]
    half=time_width//2;rotary=width//heads//2
    ops[:0]=[{'id':'time_features','op':'sinusoidal_features_checked','kernel':'sinusoidal_features_f32_checked',
        'returns_status':True,'weight_refs':{'frequencies':'timestep.inverse_frequency'},
        'graph_slots':{'inputs':{'positions':'external:time'},'outputs':{'output':'time_features'}},
        'params':{'R':1,'C':half,'scale':1000.,'call_constants':{'sinusoid_position_elements':1,
            'sinusoid_frequency_elements':half,'sinusoid_output_elements':time_width,'sinusoid_rows':1,'sinusoid_columns':half}}},
        {'id':'rotary_tables','op':'sinusoidal_constant_tables_checked','kernel':'sinusoidal_constant_tables_f32_checked',
        'returns_status':True,'weight_refs':{'positions':'rotary.positions','frequencies':'rotary_embed.inv_freq'},
        'graph_slots':{'inputs':{},'outputs':{'sine':'sine','cosine':'cosine'}},
        'params':{'R':3,'C':rotary,'scale':1.,'call_constants':{'sinusoid_position_elements':3,
            'sinusoid_frequency_elements':rotary,'sinusoid_sine_elements':3*rotary,
            'sinusoid_cosine_elements':3*rotary,'sinusoid_rows':3,'sinusoid_columns':rotary}}}]
    add=copy.deepcopy(block['block_types']['block']['body']['ops'][8]);add.update(id='conditioning')
    add['graph_slots']={'inputs':{'left':'time_emb','right':'condition_emb'},'outputs':{'output':'conditioning'}}
    add['params']['R']=1
    for key in ('sum_left_elements','sum_right_elements','sum_output_elements'):add['params']['call_constants'][key]=width
    add['params']['call_constants']['sum_rows']=1;ops.append(add)
    def concat(name,left,right,l,r):
        return {'id':name,'op':'audio_concat_channels_checked','kernel':'audio_concat_channels_checked_f32',
            'returns_status':True,'graph_slots':{'inputs':{'left':left,'right':right},'outputs':{'output':name}},
            'params':{'L':l,'R':r,'O':l+r,'T':width,'call_constants':{'concat_left_elements':l*width,
                'concat_right_elements':r*width,'concat_output_elements':(l+r)*width,
                'concat_left_stride':width,'concat_right_stride':width,'concat_output_stride':width,
                'concat_left_channels':l,'concat_right_channels':r,'concat_output_channels':l+r,'concat_frames':width}}}
    ops.extend([concat('pair','history_emb','latent_emb',1,1),concat('joined','conditioning','pair',1,2)])
    previous='joined'
    for layer in range(depth):
        prefix=f'layer{layer}_'
        for name,buffer in block['activation_buffers'].items():
            if name not in ('input','cosine','sine'):shapes[prefix+name]=buffer['shape']
        def renamed(value):
            if value=='external:input':return previous
            if value in ('external:cosine','external:sine'):return value.split(':',1)[1]
            return prefix+value
        for original in block['block_types']['block']['body']['ops']:
            op=copy.deepcopy(original);op['id']=prefix+op['id']
            for direction,ports in op['graph_slots'].items():op['graph_slots'][direction]={key:renamed(value) for key,value in ports.items()}
            if 'weight_refs' in op:op['weight_refs']={key:f'blocks.{layer}.'+value for key,value in op['weight_refs'].items()}
            ops.append(op)
        previous=prefix+'output'
    norm=copy.deepcopy(block['block_types']['block']['body']['ops'][0]);norm['id']='final_norm'
    norm['graph_slots']={'inputs':{'input':previous},'outputs':{'output':'final_norm'}}
    norm['weight_refs']={'gamma':'final_layer.norm_final.weight'};ops.append(norm)
    ops.append(linear('output','final_norm','output',width,latent,3,'final_layer.linear'))
    graph['activation_buffers']={name:{'shape':shape} for name,shape in shapes.items()}
    graph['activation_bindings']={name:name for name in shapes}
    graph['block_types']['block']['body']['ops']=ops
    graph['semantic_checkpoints']['exports']={op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
        'checkpoints':[{'id':'ming3.dit_full.'+op['id']+'.'+port,'producer':op['id'],
            'tensor':tensor,'logical_layout':'row_major_contiguous','axis_names':['row','channel'],
            'storage_dtype':'fp32'} for port,tensor in op['graph_slots']['outputs'].items()]} for op in ops}
    graph['execution_exports']={'schema':'cke.checked.execution_exports.v1',
        'entries':{'run':graph['native_entry']['function']},
        'ports':{name:execution_port(graph,name) for name in ('latent','time','condition','history')}}
    graph['execution_exports']['ports']['prediction']=execution_port(graph,'output',leading_slice=(2,1))
    return graph

def constant_weights(time_width=256):
    """Compile-time mathematical constants, never oracle activation captures."""
    import math
    import numpy as np
    if type(time_width) is not int or time_width<4 or time_width>1024 or time_width%2:
        raise ValueError('invalid timestep width')
    half=time_width//2
    inverse=np.exp(np.arange(half,dtype=np.float32)*np.float32(-math.log(10000)/(half-1)))
    return {'timestep.inverse_frequency':inverse,'rotary.positions':np.arange(3,dtype=np.float32)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--check',action='store_true');a=p.parse_args();content=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if a.output.read_text()!=content:raise SystemExit('Ming DiT circuit stale')
    else:a.output.write_text(content)
if __name__=='__main__':main()
