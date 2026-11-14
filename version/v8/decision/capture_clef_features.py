"""Development-only pinned PyTorch backbone capture for CKE head replay.

Local assets only. No downloads, deployed model scheduler, or claim of a CKE
backbone. Reject overlength requests rather than accepting reference truncation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from server.decision import contract as wire

ROOT=Path(__file__).resolve().parents[3]
IDENTITY=ROOT/'tests/fixtures/decision/clef_flash/identity.json'


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def strict_encode(reference,tokenizer,record,max_tokens):
    """Use the pinned schema encoder unchanged; detect its truncation first."""
    if type(max_tokens) is not int or not 1<=max_tokens<=4096:raise ValueError('unsupported token capacity')
    if type(record) is not dict or set(record)-{'id','model','state','questions'}:raise ValueError('text-only request required')
    if 'state' not in record or not isinstance(record.get('questions'),dict) or not 1<=len(record['questions'])<=16:raise ValueError('invalid request')
    for ident,q in record['questions'].items():
        if not wire.matches(wire.IDENT,ident) or not isinstance(q,dict) or q.get('type') not in reference.QUESTION_TYPES:raise ValueError('invalid question')
        options=reference.question_options(q)
        if not 1<=len(options)<=16 or len({x[0] for x in options})!=len(options):raise ValueError('invalid options')
        if not all(wire.matches(wire.IDENT,x[0]) for x in options):raise ValueError('unsupported candidate identity')
    # All input is bounded before encode; this large temporary limit prevents
    # the reference slicing state_ids. It does not allocate a tensor of this size.
    if len(json.dumps(record).encode())>16384:raise ValueError('request too large')
    full=reference.encode_record(tokenizer,record,max_length=2**31-1)
    if len(full.input_ids)>max_tokens:raise ValueError('request would truncate state or schema')
    bounded=reference.encode_record(tokenizer,record,max_length=max_tokens)
    if full!=bounded:raise ValueError('bounded encoding differs')
    return bounded


def compact_inputs(encoded,hidden,rows,original_token_ids):
    """Retain exact gathered embedding rows; remap only head-local token IDs."""
    ids=np.asarray(encoded.input_ids,dtype=np.int64)
    unique=np.unique(ids)
    if not np.array_equal(unique,np.asarray(original_token_ids)):raise ValueError('embedding row identities differ')
    hidden=np.asarray(hidden,dtype=np.float32);rows=np.asarray(rows,dtype=np.float32)
    if hidden.ndim!=2 or hidden.shape[0]!=len(ids) or rows.shape!=(len(unique),hidden.shape[1]):raise ValueError('feature shape mismatch')
    if not np.isfinite(hidden).all() or not np.isfinite(rows).all():raise ValueError('nonfinite features')
    questions=encoded.questions
    spans=[s for q in questions for s in q.option_spans]
    for a,b in [q.question_span for q in questions]+spans:
        if not 0<=a<b<=len(ids):raise ValueError('invalid span')
    return dict(hidden=hidden,input_ids=np.searchsorted(unique,ids).astype('i4'),
                question_spans=np.array([q.question_span for q in questions],dtype='i4'),
                option_spans=np.array(spans,dtype='i4'),global_span=np.array([[len(ids)-1,len(ids)]],dtype='i4'),
                option_counts=np.array([len(q.option_spans) for q in questions],dtype='i4'),
                question_type_ids=np.array([q.question_type for q in questions],dtype='i4'))


def verify_release(folder):
    """Verify pinned metadata/head/code and retain actual local backbone hashes.

    Backbone hashes are measured provenance, not an assertion that arbitrary
    shards belong to the publisher revision. A trusted release lock is required.
    """
    identity=json.loads(IDENTITY.read_text());folder=Path(folder)
    pairs={'config.json':'config_sha256','joint_head_config.json':'joint_head_config_sha256',
           'joint_schema_model.py':'joint_schema_model_sha256','joint_head.safetensors':'joint_head_weights_sha256'}
    for name,key in pairs.items():
        if file_sha(folder/name)!=identity[key]:raise ValueError('pinned release identity mismatch: '+name)
    return identity


def verify_lock(folder,lock_path,expected_sha):
    """Administrator-pinned list of release files; verify before importing code."""
    if file_sha(lock_path)!=expected_sha:raise ValueError('release lock identity mismatch')
    lock=json.loads(Path(lock_path).read_text());identity=json.loads(IDENTITY.read_text())
    if lock.get('model')!=identity['model'] or lock.get('revision')!=identity['revision']:raise ValueError('release lock revision mismatch')
    files=lock.get('files',{})
    required={'config.json','joint_head_config.json','joint_schema_model.py','joint_head.safetensors','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}
    if not isinstance(files,dict) or not required<=files.keys():raise ValueError('incomplete release lock')
    for name,expected in files.items():
        if Path(name).name!=name or name in ('.','..') or file_sha(Path(folder)/name)!=expected:raise ValueError('release file mismatch')
    # A local loader must not discover an alternative unpinned weight/tokenizer file.
    discoverable={p.name for p in Path(folder).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.py','.txt','.model','.jinja') and p.resolve()!=Path(lock_path).resolve()}
    if not discoverable<=files.keys():raise ValueError('unlocked loader assets')
    index=json.loads((Path(folder)/'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values())<=files.keys():raise ValueError('unlocked model shards')
    return lock


def capture(folder,lock_path,lock_sha,request_path,output,*,max_tokens=512,threads=4,attention="eager",schedule="prefill"):
    if attention not in ("eager", "sdpa"):raise ValueError("unsupported reference attention backend")
    if schedule not in ("prefill", "incremental"):raise ValueError("unsupported reference schedule")
    import torch
    import transformers
    import tokenizers
    from safetensors.torch import load_file
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    folder=Path(folder);output=Path(output)
    if output.exists():raise ValueError('output already exists')
    lock=verify_lock(folder,lock_path,lock_sha);identity=verify_release(folder)
    spec=importlib.util.spec_from_file_location('cke_pinned_clef_capture',folder/'joint_schema_model.py')
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    record=wire.decode(Path(request_path).read_bytes());encoded=strict_encode(reference,tokenizer,record,max_tokens)
    if type(threads) is not int or not 1<=threads<=64:raise ValueError('invalid thread count')
    torch.set_num_threads(threads)
    backbone=Qwen3_5ForConditionalGeneration.from_pretrained(folder,dtype=torch.bfloat16,local_files_only=True,use_safetensors=True,attn_implementation=attention).eval()
    text_model=backbone.model
    if hasattr(text_model,'language_model'):text_model=text_model.language_model
    ids=torch.tensor(encoded.input_ids,dtype=torch.int64)[None]
    unique=torch.unique(ids,sorted=True)
    with torch.inference_mode():
        if schedule == "prefill":
            hidden=text_model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state[0].float().cpu().numpy()
        else:
            cache=None;features=[]
            for position in range(ids.shape[1]):
                step=text_model(input_ids=ids[:,position:position+1],attention_mask=torch.ones_like(ids[:,:position+1]),past_key_values=cache,use_cache=True,return_dict=True)
                cache=step.past_key_values
                features.append(step.last_hidden_state[0].float().cpu().numpy().copy())
            hidden=np.concatenate(features,axis=0)
            del cache,step

        rows=backbone.get_output_embeddings().weight[unique].float().cpu().numpy()
    inputs=compact_inputs(encoded,hidden,rows,unique.cpu().numpy())
    # Free the large backbone before building the FP32 head reference. Head
    # replay intentionally compares identical FP32-cast backbone operands.
    del text_model,backbone
    head=reference.JointSchemaHead(**json.loads((folder/'joint_head_config.json').read_text()))
    head.load_state_dict(load_file(folder/'joint_head.safetensors'),strict=True);head.float().eval()
    from dataclasses import replace
    compact_record=replace(encoded,input_ids=tuple(map(int,inputs['input_ids'])))
    checkpoints={};hooks=[]
    for i,layer in enumerate(head.layers):
        hooks.append(layer.register_forward_hook(lambda m,a,y,i=i:checkpoints.__setitem__(f'fields{i}_output',y.detach().float().cpu().numpy().reshape(-1,y.shape[-1]))))
    hooks.append(head.field_norm.register_forward_hook(lambda m,a,y:checkpoints.__setitem__('final_fields',y.detach().float().cpu().numpy().reshape(-1,y.shape[-1]))))
    try:
        with torch.inference_mode():
            logits=head(torch.from_numpy(hidden)[None],torch.from_numpy(inputs['input_ids'].astype('int64'))[None],torch.ones_like(ids),[compact_record],torch.from_numpy(rows))[0]
        checkpoints['option_logits']=torch.cat(logits).cpu().numpy().reshape(-1,1)
    finally:
        for hook in hooks:hook.remove()
    output.mkdir(parents=True)
    np.savez(output/'features.npz',**inputs);np.save(output/'embedding.npy',rows);np.savez(output/'reference.npz',**checkpoints)
    np.save(output/'original_token_ids.npy',ids.cpu().numpy()[0]);np.save(output/'embedding_token_ids.npy',unique.cpu().numpy())
    questions=[dict(id=q.question_id,type=q.question_type,span=list(q.question_span),option_spans=[list(s) for s in q.option_spans],candidate_ids=list(q.option_ids)) for q in encoded.questions]
    metadata=dict(schema='cke.clef.feature-capture.v1',model=identity['model'],revision=identity['revision'],release_lock_sha256=lock_sha,
                  request_sha256=file_sha(request_path),reference_source_sha256=identity['joint_schema_model_sha256'],head_sha256=identity['joint_head_weights_sha256'],
                  backbone_executor='pytorch_reference',backbone_dtype='bf16',head_dtype='fp32',attention=attention,backbone_schedule=schedule,torch_version=torch.__version__,transformers_version=transformers.__version__,tokenizers_version=tokenizers.__version__,threads=threads,
                  tokens=len(encoded.input_ids),questions=questions,files={p.name:file_sha(p) for p in sorted(output.iterdir())},
                  evidence='captured reference backbone; CKE backbone NOT_TESTED; same-input FP32 head oracle')
    (output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-lock',type=Path,required=True);p.add_argument('--release-lock-sha256',required=True)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-tokens',type=int,default=512);p.add_argument('--threads',type=int,default=4);p.add_argument('--attention',choices=['eager','sdpa'],default='eager');p.add_argument('--schedule',choices=['prefill','incremental'],default='prefill')
    a=p.parse_args();m=capture(a.release,a.release_lock,a.release_lock_sha256,a.request,a.output,max_tokens=a.max_tokens,threads=a.threads,attention=a.attention,schedule=a.schedule)
    print(json.dumps(dict(status='CAPTURED',tokens=m['tokens'],backbone_executor=m['backbone_executor'],capture_sha256=file_sha(a.output/'capture.json'))))

if __name__=='__main__':main()
