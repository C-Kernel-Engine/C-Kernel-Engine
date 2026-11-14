#!/usr/bin/env python3
"""Opt-in image-answer and follow-up acceptance for one pinned HTTP bundle.

This is task/protocol acceptance, not an independent numerical oracle. Each run
gets a new report directory; failed and incomplete attempts are retained.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
import uuid


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def assess(response, expected):
    if not isinstance(response,dict) or response.get('object')!='chat.completion':
        raise ValueError('not a chat completion')
    if not isinstance(response.get('id'),str) or not response['id']:
        raise ValueError('missing request ID')
    choices=response.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or choices[0].get('finish_reason')!='stop':
        raise ValueError('request did not finish normally')
    message=choices[0].get('message',{})
    if message.get('role')!='assistant' or not isinstance(message.get('content'),str):
        raise ValueError('invalid assistant message')
    answer=json.loads(message['content'],object_pairs_hook=unique_object)
    if not isinstance(answer,dict) or set(answer)!=set(expected):
        raise ValueError('answer does not match requested schema')
    if any(not isinstance(answer[k],str) or answer[k].casefold()!=v for k,v in expected.items()):
        raise ValueError('incorrect image answer')
    usage=response.get('usage',{})
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens','total_tokens')):
        raise ValueError('invalid usage')
    if usage['prompt_tokens']+usage['completion_tokens']!=usage['total_tokens']:
        raise ValueError('inconsistent usage')
    return message['content']


def request(base, route, body=None, timeout=300):
    req=urllib.request.Request(base.rstrip('/')+route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as reply:
        raw=reply.read(2*1024*1024+1)
    if len(raw)>2*1024*1024: raise ValueError('oversized response')
    return json.loads(raw,object_pairs_hook=unique_object)


def identity(base):
    data=request(base,'/v1/capabilities',timeout=10)
    for key in ('server_instance_id','bundle_identity'):
        if not isinstance(data.get(key),str) or not data[key]: raise ValueError('missing '+key)
    if not data.get('loaded_libraries'):raise ValueError('missing loaded library identities')
    return data


def run(base, model, fixtures, output, expected_bundle, timeout=300, require_reference=False):
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'cke.vision.http.acceptance.v1','status':'INCOMPLETE',
        'started_at':datetime.now(timezone.utc).isoformat(),'cases':[],
        'numerical_oracle':'NOT_TESTED','harness_task':'NOT_TESTED'}
    target=output/'report.json';write(target,report)
    try:
        initial=identity(base);report['identity']=initial
        if initial['bundle_identity']!=expected_bundle:raise ValueError('unexpected bundle')
        ids=set()
        sequence=[('red-blue.png',{'left':'red','right':'blue'}),
                  ('blue-red.png',{'left':'blue','right':'red'}),
                  ('red-blue.jpg',{'left':'red','right':'blue'}),
                  ('red-blue.png',{'left':'red','right':'blue'})]
        for filename,expected in sequence:
            with (fixtures/filename).open('rb') as source:image=source.read(8*1024*1024+1)
            if len(image)>8*1024*1024:raise ValueError('fixture exceeds limit')
            mime='image/jpeg' if filename.endswith('.jpg') else 'image/png'
            messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(image).decode()}},
                {'type':'text','text':'Identify the left and right colors. Return only JSON with keys left and right, using plain color names. No markdown.'}]}]
            for turn in ('initial','followup'):
                case={'fixture':filename,'turn':turn,'image_sha256':hashlib.sha256(image).hexdigest(),'status':'INCOMPLETE'}
                report['cases'].append(case);write(target,report)
                started=time.monotonic()
                response=request(base,'/v1/chat/completions',{'model':model,'messages':messages,'max_tokens':96},timeout)
                case['seconds']=time.monotonic()-started;case['response']=response
                write(target,report)
                answer=assess(response,expected)
                if response['id'] in ids:raise ValueError('reused request ID')
                ids.add(response['id'])
                evidence=response.get('cke_evidence',{})
                if require_reference and evidence.get('direct_reference_match') is not True:
                    raise ValueError('missing direct generated reference equality')
                if evidence.get('bundle_identity')!=expected_bundle or evidence.get('image_sha256')!=case['image_sha256']:
                    raise ValueError('response artifact/image identity mismatch')
                current=identity(base)
                if any(current[k]!=initial[k] for k in ('server_instance_id','bundle_identity','loaded_libraries')):
                    raise ValueError('service changed during acceptance')
                case['status']='PASS';write(target,report)
                messages.extend([{'role':'assistant','content':answer},
                    {'role':'user','content':'Check the same image again. Return the same two JSON keys with the correct left and right colors.'}])
        report['status']='PASS'
        report['same_artifact_initial_logits']='PASS' if require_reference else 'NOT_TESTED'
    except (TimeoutError,socket.timeout) as error:
        report['status']='INCOMPLETE';report['error']=type(error).__name__
    except urllib.error.URLError as error:
        report['status']='INCOMPLETE' if isinstance(error.reason,(TimeoutError,socket.timeout)) else 'FAIL'
        report['error']=str(error)
    except Exception as error:
        report['status']='FAIL';report['error']=str(error)
    finally:
        if report['cases'] and report['cases'][-1]['status']=='INCOMPLETE' and report['status']=='FAIL':
            report['cases'][-1]['status']='FAIL'
        report['finished_at']=datetime.now(timezone.utc).isoformat();write(target,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8120')
    parser.add_argument('--model',required=True);parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--expected-bundle',required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--require-reference',action='store_true')
    args=parser.parse_args()
    if args.timeout<=0:parser.error('timeout must be positive')
    output=args.output_root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    report=run(args.base_url,args.model,args.fixtures,output,args.expected_bundle,args.timeout,args.require_reference)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json')}))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
