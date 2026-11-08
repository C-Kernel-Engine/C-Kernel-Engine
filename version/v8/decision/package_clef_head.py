"""Offline packaging of a bounded generated Clef head; no reference arithmetic.

CLI imports safetensors/numpy for weight and feature preparation.
The deployed worker requires neither. Inputs are supplied backbone features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

# Existing authoring modules are also executable scripts with sibling imports.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from version.v8.decision.build_clef_flash_complete_head_circuit import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

ROOT=Path(__file__).resolve().parents[3]
INPUTS=('hidden','input_ids','question_spans','option_spans','global_span','option_counts','question_type_ids')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def effective_weights(graph, raw):
    """Split packed QKV and convert effective weights to FP32; no inference arithmetic."""
    result=dict(raw)
    for i in range(4):
        for attention in ('self_attn','multihead_attn'):
            stem=f'layers.{i}.{attention}'
            w=raw[stem+'.in_proj_weight']; b=raw[stem+'.in_proj_bias']; width=w.shape[1]
            for j,axis in enumerate(('q','k','v')):
                result[f'{stem}.{axis}_proj.weight']=w[j*width:(j+1)*width].copy()
                result[f'{stem}.{axis}_proj.bias']=b[j*width:(j+1)*width].copy()
    for i in range(2):
        stem=f'evidence_layers.{i}.attention'
        if stem+'.in_proj_weight' in raw:
            w=raw[stem+'.in_proj_weight']; b=raw[stem+'.in_proj_bias']; width=w.shape[1]
            for j,axis in enumerate(('q','k','v')):
                result[f'{stem}.{axis}_proj.weight']=w[j*width:(j+1)*width].copy()
                result[f'{stem}.{axis}_proj.bias']=b[j*width:(j+1)*width].copy()
    needed={v for block in graph['block_types'].values()
            for op in block.get('header',[])+block.get('body',{}).get('ops',[])+block.get('footer',[])
            for v in op.get('weight_refs',{}).values()}
    return {k:np.ascontiguousarray(result[k],dtype='f4') for k in sorted(needed)}


def build_package(folder,raw,artifacts,model,*,heads,provenance):
    """Compile once, serialize weight spans and registered supplied-feature inputs.

    artifacts: [{id, inputs: numpy arrays, question_index, candidate_ids}].
    Parity is never inferred here; the HTTP evidence remains NOT_TESTED.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    first=artifacts[0]['inputs'];tokens,channels=first['hidden'].shape
    graph=build_circuit(tokens=tokens,questions=len(first['question_spans']),options=len(first['option_spans']),channels=channels,width=raw['question_projection.weight'].shape[0],heads=heads,feedforward=raw['layers.0.linear1.weight'].shape[0],vocabulary=len(raw['output_embedding.weight']))
    weights=effective_weights(graph,raw);entries=[];payload=bytearray()
    for name,value in sorted(weights.items()):
        payload.extend(b'\0'*((-len(payload))&63));offset=len(payload)
        data=np.ascontiguousarray(value,dtype='<f4').tobytes();payload.extend(data)
        entries.append(dict(name=name,dtype='fp32',shape=list(value.shape),file_offset=offset,size=len(data),sha256=sha(data)))
    (folder/'weights.bin').write_bytes(payload)
    integer_names=set(INPUTS)-{'hidden'}|{'option_row_indices','option_question_indices'}
    source=dict(config=dict(model=graph['name'],arch=graph['name'],num_layers=1,embed_dim=channels,num_heads=1,num_kv_heads=1,head_dim=channels,intermediate_size=channels,context_length=tokens,max_seq_len=tokens,vocab_size=len(raw['output_embedding.weight']),epsilon=1e-5,activation_buffer_dtypes={n:'i32' for n in integer_names}),entries=entries,quant_summary={},template=graph)
    (folder/'circuit.json').write_bytes(canonical(graph))
    layout,calls,library,loaded,execute=compile_native_graph(folder,source,graph)
    if calls['errors']:raise ValueError('compiler rejected package')
    planned={e['name']:e for e in layout['memory']['weights']['entries']}
    weight_spans=[dict(name=e['name'],arena_offset=planned[e['name']]['abs_offset'],file_offset=e['file_offset'],size=e['size']) for e in entries]
    acts={e['name']:e for e in layout['memory']['activations']['buffers']}
    inputs={n:dict(offset=acts[n]['abs_offset'],shape=graph['activation_buffers'][n]['shape'],dtype='fp32' if n=='hidden' else 'i32') for n in INPUTS}
    records=[]
    for i,artifact in enumerate(artifacts):
        data=bytearray();spans=[]
        if set(artifact['inputs'])!=set(INPUTS):raise ValueError('input names mismatch')
        for name,spec in inputs.items():
            array=artifact['inputs'][name]
            if list(array.shape)!=spec['shape'] or array.dtype!=np.dtype('f4' if name=='hidden' else 'i4'):raise ValueError('input shape/dtype mismatch')
            encoded=np.ascontiguousarray(array,dtype='<f4' if name=='hidden' else '<i4').tobytes()
            spans.append(dict(name=name,file_offset=len(data),size=len(encoded)));data.extend(encoded)
        filename=f'features-{i}.bin';(folder/filename).write_bytes(data)
        record=dict(id=artifact['id'],file=filename,sha256=sha(data),spans=spans,question_index=artifact['question_index'],candidate_ids=artifact['candidate_ids'])
        record['input_sha256']=sha(canonical(record));records.append(record)
    files={key:dict(file=filename,sha256=sha((folder/filename).read_bytes())) for key,filename in [('library','generated.so'),('layout','layout.json'),('ir','call.json'),('circuit','circuit.json'),('weights','weights.bin'),('build_identity','build-identity.json'),('generated_source','generated.c')]}
    manifest=dict(schema='cke.decision.head-package.v1',model=model,cke_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),tokenizer_revision=None,provenance=provenance,files=files,function=graph['native_entry']['function'],arena_bytes=layout['memory']['arena']['total_size'],weight_spans=weight_spans,inputs=inputs,output=dict(offset=acts['option_logits']['abs_offset'],count=len(first['option_spans'])),vocabulary=len(raw['output_embedding.weight']),artifacts=records)
    encoded=canonical(manifest);(folder/'package.json').write_bytes(encoded)
    return sha(encoded)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',type=Path,required=True);p.add_argument('--head-sha256',required=True)
    p.add_argument('--embedding',type=Path,required=True);p.add_argument('--embedding-sha256',required=True)
    p.add_argument('--registrations',type=Path,required=True,help='Local JSON list: id, npz, question_index, candidate_ids')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',required=True);p.add_argument('--heads',type=int,default=16)
    args=p.parse_args()
    from safetensors.numpy import load_file
    if sha(args.head.read_bytes())!=args.head_sha256 or sha(args.embedding.read_bytes())!=args.embedding_sha256:raise ValueError('weight identity mismatch')
    raw=load_file(str(args.head));raw['output_embedding.weight']=np.load(args.embedding,allow_pickle=False)
    artifacts=[]
    for item in json.loads(args.registrations.read_text()):
        with np.load(item['npz'],allow_pickle=False) as data:inputs={n:data[n].copy() for n in INPUTS}
        artifacts.append(dict(id=item['id'],inputs=inputs,question_index=item['question_index'],candidate_ids=item['candidate_ids']))
    digest=build_package(args.output,raw,artifacts,dict(id='clef-flash-head',revision=args.revision),heads=args.heads,provenance=dict(head_weights_sha256=args.head_sha256,embedding_file_sha256=args.embedding_sha256,features='supplied',embedding='supplied'))
    print(json.dumps(dict(package_sha256=digest,implementation_stage='head_only',parity='NOT_TESTED')))
if __name__=='__main__':main()
