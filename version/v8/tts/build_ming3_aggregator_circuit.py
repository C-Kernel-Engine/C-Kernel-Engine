#!/usr/bin/env python3
"""Generated latent-patch aggregator feeding the next backbone embedding."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit as block_circuit
from version.v8.scripts.checked_circuit_authoring_v8 import linear
OUTPUT=ROOT/'version/v8/circuits/ming3_aggregator_fp32_bounded.json'

def build_circuit(patches=1,width=1024,heads=16,depth=8,latent=64,output_width=896,mult=4):
    if any(type(v) is not int or v<1 for v in (patches,depth,latent,output_width)) or patches>16 or depth>8 or latent>256 or output_width>4096:
        raise ValueError('unsupported aggregator geometry')
    t=patches+1;block=block_circuit(t,width,heads,mult);g=copy.deepcopy(block)
    g['name']=f'ming3_aggregator_fp32_bounded_d{depth}'
    g['native_entry']['function']='ck_ming3_aggregator_fp32_bounded'
    g['contract']['runtime_invariants'].update(complete_aggregator=True,supplied_rotary_tables=False,
        patches=patches,depth=depth,latent=latent,output_width=output_width,class_id=0,
        scope='FP32 latent patch to backbone embedding; not publisher BF16 parity')
    shapes={'latent':[patches,latent],'class_id':[1],'class_embedding':[1,width],'embedded':[patches,width],
            'joined':[t,width],'cosine':[t,width//heads//2],'sine':[t,width//heads//2],
            'final_norm':[t,width],'all_rows':[t,output_width],'output':[1,output_width]}
    ops=[{'id':'class_embedding','op':'embedding_lookup_token_major_checked','kernel':'audio_text_embedding_token_major_f32',
        'returns_status':True,'weight_refs':{'table':'word_embedder.weight'},
        'graph_slots':{'inputs':{'ids':'external:class_id'},'outputs':{'output':'class_embedding'}},
        'params':{'T':1,'V':1,'C':width,'call_constants':{'id_elements':1,'table_elements':width,
            'vocabulary':1,'channels':width,'table_stride':width,'output_elements':width,'tokens':1,'output_stride':width}}},
        linear('embedded','external:latent','embedded',latent,width,patches,'x_embedder'),
        {'id':'joined','op':'audio_concat_channels_checked','kernel':'audio_concat_channels_checked_f32','returns_status':True,
         'graph_slots':{'inputs':{'left':'class_embedding','right':'embedded'},'outputs':{'output':'joined'}},
         'params':{'L':1,'R':patches,'O':t,'T':width,'call_constants':{'concat_left_elements':width,
            'concat_right_elements':patches*width,'concat_output_elements':t*width,'concat_left_stride':width,
            'concat_right_stride':width,'concat_output_stride':width,'concat_left_channels':1,
            'concat_right_channels':patches,'concat_output_channels':t,'concat_frames':width}}},
        {'id':'rotary_tables','op':'sinusoidal_constant_tables_checked','kernel':'sinusoidal_constant_tables_f32_checked',
         'returns_status':True,'weight_refs':{'positions':'rotary.positions','frequencies':'rotary_embed.inv_freq'},
         'graph_slots':{'inputs':{},'outputs':{'sine':'sine','cosine':'cosine'}},
         'params':{'R':t,'C':width//heads//2,'scale':1.,'call_constants':{
             'sinusoid_position_elements':t,'sinusoid_frequency_elements':width//heads//2,
             'sinusoid_sine_elements':t*width//heads//2,'sinusoid_cosine_elements':t*width//heads//2,
             'sinusoid_rows':t,'sinusoid_columns':width//heads//2}}}]
    previous='joined'
    for layer in range(depth):
        prefix=f'layer{layer}_'
        for name,b in block['activation_buffers'].items():
            if name not in ('input','cosine','sine'):shapes[prefix+name]=b['shape']
        def rename(value):
            if value=='external:input':return previous
            if value in ('external:cosine','external:sine'):return value.split(':')[1]
            return prefix+value
        for original in block['block_types']['block']['body']['ops']:
            op=copy.deepcopy(original);op['id']=prefix+op['id']
            op['graph_slots']={direction:{k:rename(v) for k,v in ports.items()} for direction,ports in op['graph_slots'].items()}
            if 'weight_refs' in op:op['weight_refs']={k:f'blocks.{layer}.'+v for k,v in op['weight_refs'].items()}
            ops.append(op)
        previous=prefix+'output'
    norm=copy.deepcopy(block['block_types']['block']['body']['ops'][0]);norm['id']='final_norm'
    norm['graph_slots']={'inputs':{'input':previous},'outputs':{'output':'final_norm'}}
    norm['weight_refs']={'gamma':'final_layer.norm_final.weight'};ops.append(norm)
    ops.append(linear('all_rows','final_norm','all_rows',width,output_width,t,'final_layer.linear'))
    ops.append({'id':'output','op':'runtime_copy_valid','kernel':'runtime_copy_valid_f32','returns_status':True,
        'graph_slots':{'inputs':{'input':'all_rows'},'outputs':{'output':'output'}},
        'params':{'C':1,'A_capacity':output_width,'call_constants':{'expanded_elements':t*output_width,
            'channels':1,'expanded_frames':output_width,'expanded_stride':output_width,
            'output_elements':output_width,'output_stride':output_width}}})
    g['activation_buffers']={k:{'shape':v} for k,v in shapes.items()};g['activation_bindings']={k:k for k in shapes}
    g['block_types']['block']['body']['ops']=ops
    g['semantic_checkpoints']['exports']={op['id']:{'section':'body','template_op_id':op['id'],'op':op['op'],
        'checkpoints':[{'id':'ming3.aggregator.'+op['id']+'.'+port,'producer':op['id'],'tensor':tensor,
                       'logical_layout':'row_major_contiguous','axis_names':['row','channel'],'storage_dtype':'fp32'}
                       for port,tensor in op['graph_slots']['outputs'].items()]} for op in ops}
    g['execution_exports']={'schema':'cke.checked.execution_exports.v1',
        'entries':{'run':g['native_entry']['function']},'ports':{
            'latent':execution_port(g,'latent'),'class_id':execution_port(g,'class_id'),
            'embedding':execution_port(g,'output')}}
    return g

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();content=json.dumps(build_circuit(),indent=2)+'\n'
    if a.check:
        if OUTPUT.read_text()!=content:raise SystemExit('aggregator circuit stale')
    else:OUTPUT.write_text(content)
if __name__=='__main__':main()
