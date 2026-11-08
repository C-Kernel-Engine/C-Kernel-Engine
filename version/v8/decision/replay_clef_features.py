"""Validate a retained reference-backbone capture against the packaged CKE head."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from server.decision.native import NativeHead
from server.decision.consumer import validate
from version.v8.decision.capture_clef_features import file_sha,IDENTITY
from version.v8.decision.package_clef_head import build_package


def load_capture(folder,expected_sha):
    folder=Path(folder)
    if file_sha(folder/'capture.json')!=expected_sha:raise ValueError('capture identity mismatch')
    m=json.loads((folder/'capture.json').read_text());pin=json.loads(IDENTITY.read_text())
    if m.get('schema')!='cke.clef.feature-capture.v1' or m.get('model')!=pin['model'] or m.get('revision')!=pin['revision']:raise ValueError('capture model mismatch')
    if m.get('reference_source_sha256')!=pin['joint_schema_model_sha256'] or m.get('head_sha256')!=pin['joint_head_weights_sha256']:raise ValueError('capture experiment mismatch')
    if m.get('backbone_executor')!='pytorch_reference' or m.get('backbone_dtype')!='bf16' or m.get('head_dtype')!='fp32':raise ValueError('unsupported capture contract')
    required={'features.npz','embedding.npy','reference.npz','original_token_ids.npy','embedding_token_ids.npy'}
    if set(m.get('files',{}))!=required:raise ValueError('capture files differ')
    for name,sha in m['files'].items():
        if file_sha(folder/name)!=sha:raise ValueError('capture file mismatch')
    with np.load(folder/'features.npz',allow_pickle=False) as f:inputs={k:f[k].copy() for k in f.files}
    with np.load(folder/'reference.npz',allow_pickle=False) as f:reference={k:f[k].copy() for k in f.files}
    rows=np.load(folder/'embedding.npy',allow_pickle=False);tokens=np.load(folder/'original_token_ids.npy',allow_pickle=False);vocab=np.load(folder/'embedding_token_ids.npy',allow_pickle=False)
    if not np.array_equal(np.unique(tokens),vocab):raise ValueError('embedding row map differs')
    ids=inputs['input_ids']
    if ids.dtype!=np.int32 or np.any(ids<0) or np.any(ids>=len(vocab)) or not np.array_equal(vocab[ids],tokens):raise ValueError('token remapping differs')
    if len(tokens)!=m['tokens'] or rows.shape!=(len(vocab),inputs['hidden'].shape[1]):raise ValueError('capture dimensions differ')
    questions=m['questions']
    if not np.array_equal(inputs['question_spans'],[q['span'] for q in questions]) or not np.array_equal(inputs['option_spans'],[s for q in questions for s in q['option_spans']]):raise ValueError('span identities differ')
    if not np.array_equal(inputs['option_counts'],[len(q['candidate_ids']) for q in questions]) or not np.array_equal(inputs['question_type_ids'],[q['type'] for q in questions]):raise ValueError('question identities differ')
    if set(reference)!={*(f'fields{i}_output' for i in range(4)),'final_fields','option_logits'}:raise ValueError('missing reference checkpoints')
    nq=len(questions);no=sum(len(q['candidate_ids']) for q in questions)
    for name,array in reference.items():
        shape=(no,1) if name=='option_logits' else (nq,1024)
        if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError('invalid reference shape or values')
    return m,inputs,rows,reference


def replay(capture_path,capture_sha,head_path,output):
    from safetensors.numpy import load_file
    m,inputs,rows,reference=load_capture(capture_path,capture_sha)
    if file_sha(head_path)!=m['head_sha256']:raise ValueError('head weights differ')
    raw=load_file(str(head_path));raw['output_embedding.weight']=rows
    artifacts=[dict(id=f'question-{i}',inputs=inputs,question_index=i,candidate_ids=q['candidate_ids']) for i,q in enumerate(m['questions'])]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    package_sha=build_package(output/'package',raw,artifacts,dict(id='clef-flash-head',revision=m['revision']),heads=16,
        provenance=dict(features='pytorch_reference_backbone',embedding='pytorch_reference_output_rows',capture_sha256=capture_sha,release_lock_sha256=m['release_lock_sha256']))
    head=NativeHead(output/'package/package.json',package_sha);report=dict(scope='head_only',backbone_executor='pytorch_reference',capture_sha256=capture_sha,package_sha256=package_sha,execution='FAIL',numerical='NOT_TESTED',checks=[],requests=[])
    try:
        layout=json.loads((output/'package/layout.json').read_text());acts={x['name']:x for x in layout['memory']['activations']['buffers']}
        baseline=None
        for i in [*range(len(artifacts)),0]:
            q=head.request(f'question-{i}');status,response=head.evaluate(q)
            report['requests'].append(dict(question_id=m['questions'][i]['id'],http_status=status,response=response))
            if status!=200:break
            result=validate(response,q)
            start=int(inputs['option_counts'][:i].sum());count=int(inputs['option_counts'][i])
            actual_scores=np.array([x['logit'] for x in result['scores']]);diff=np.abs(actual_scores-reference['option_logits'][start:start+count,0])
            report['checks'].append(dict(checkpoint=f'question-{i}-returned-logits',max_abs=float(diff.max()),rmse=float(np.sqrt(np.mean(diff**2))),atol=2e-6,rtol=0,status='PASS' if np.all(diff<=2e-6) else 'FAIL'))
            if baseline is None:
                for name,want in reference.items():
                    actual=np.ctypeslib.as_array((ctypes.c_float*want.size).from_address(head.address+acts[name]['abs_offset'])).reshape(want.shape).copy()
                    diff=np.abs(actual.astype('f8')-want);finite=bool(np.isfinite(diff).all());atol=2e-6 if name=='option_logits' else 1e-5
                    report['checks'].append(dict(checkpoint=name,max_abs=float(diff.max()) if finite else None,rmse=float(np.sqrt(np.mean(diff**2))) if finite else None,atol=atol,rtol=0,status='PASS' if np.isfinite(actual).all() and np.all(diff<=atol) else 'FAIL'))
                baseline=result['scores']
            elif i==0 and result['scores']!=baseline:raise ValueError('reused request differs')
        else:
            report['execution']='PASS';report['numerical']='PASS' if all(x['status']=='PASS' for x in report['checks']) else 'FAIL'
    finally:head.close()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True);p.add_argument('--head',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=replay(a.capture,a.capture_sha256,a.head,a.output)
    print(json.dumps({k:r[k] for k in ('scope','backbone_executor','execution','numerical')}))
    if r['execution']!='PASS' or r['numerical']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
