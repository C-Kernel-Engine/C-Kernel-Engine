"""Independent simulated Antsand consumer: fail closed, retain no private prompts.

Does not import producer validators. No real Antsand transport integration.
"""
import argparse
import copy
import hashlib
import http.client
import json
import math
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES=16384
class InvalidResult(ValueError): pass

def check(ok,code='INVALID_RESPONSE'):
    if not ok:raise InvalidResult(code)

def keys(value,names):check(type(value) is dict and set(value)==set(names))
def finite(v):
    try:return type(v) in (int,float) and math.isfinite(v)
    except OverflowError:return False
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_expected(q):
    keys(q,('schema','request_id','adapter_boot_id','worker_boot_id','model','task','input','candidate_ids','timeout_ms','policy'))
    ident=lambda x:type(x) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',x) is not None
    hexid=lambda x,n:type(x) is str and re.fullmatch('[a-f0-9]{'+str(n)+'}',x) is not None
    check(q['schema']=='cke.decision.evaluate.v1' and ident(q['request_id']),'INVALID_REQUEST')
    check(hexid(q['adapter_boot_id'],32) and hexid(q['worker_boot_id'],32),'INVALID_REQUEST')
    keys(q['model'],('id','revision'));check(ident(q['model']['id']) and hexid(q['model']['revision'],'40,64'),'INVALID_REQUEST')
    keys(q['input'],('artifact_id','sha256'));check(ident(q['input']['artifact_id']) and hexid(q['input']['sha256'],64),'INVALID_REQUEST')
    check(q['task']=='registered_feature_scores','INVALID_REQUEST')
    ids=q['candidate_ids'];check(type(ids) is list and 1<=len(ids)<=16 and all(ident(x) for x in ids),'INVALID_REQUEST')
    check(len(ids)==len(set(ids)),'INVALID_REQUEST')
    check(type(q['timeout_ms']) is int and 1<=q['timeout_ms']<=1000,'INVALID_REQUEST')
    keys(q['policy'],('minimum_logit_margin',));v=q['policy']['minimum_logit_margin']
    check(finite(v) and 0<=v<=100,'INVALID_REQUEST')
    check(len(json.dumps(q,allow_nan=False).encode())<=MAX_BYTES,'MESSAGE_TOO_LARGE')

def validate(r,q,allow_simulation=False):
    validate_expected(q)
    keys(r,('schema','request_id','adapter_boot_id','worker_boot_id','model','request_sha256','ok','result','error'))
    check(r['schema']=='cke.decision.evaluate.v1')
    for key in ('request_id','adapter_boot_id','worker_boot_id','model'):
        check(r[key]==q[key],'CORRELATION_MISMATCH')
    check(r['request_sha256']==sha(q),'CORRELATION_MISMATCH')
    check(type(r['ok']) is bool)
    if not r['ok']:
        check(r['result'] is None);keys(r['error'],('code',))
        check(r['error']['code'] in ('INVALID_REQUEST','UNKNOWN_MODEL','UNKNOWN_ARTIFACT','UNSUPPORTED_TASK','CANDIDATE_MISMATCH','BUSY','DEADLINE_EXCEEDED','RESTARTED','NONFINITE_SCORE','INTERNAL_ERROR'))
        raise InvalidResult('REMOTE_'+r['error']['code'])
    check(r['error'] is None);d=r['result']
    keys(d,('implementation_stage','execution_kind','scores','score_semantics','decision','executed','latency_ms','evidence'))
    check(d['implementation_stage'] in ('contract_simulation','head_only','end_to_end'))
    check(d['execution_kind'] in ('fixture_replay','native_generated'))
    check((d['implementation_stage']=='contract_simulation')==(d['execution_kind']=='fixture_replay'))
    check(allow_simulation or d['execution_kind']=='native_generated','SIMULATION_NOT_ALLOWED')
    check(d['executed'] is False and d['score_semantics']=='raw_logits')
    check(finite(d['latency_ms']) and 0<=d['latency_ms']<=q['timeout_ms'])
    scores=d['scores'];ids=q['candidate_ids']
    check(type(scores) is list and len(scores)==len(ids))
    for item,ident in zip(scores,ids):
        keys(item,('candidate_id','logit'));check(item['candidate_id']==ident);check(finite(item['logit']))
    dec=d['decision'];keys(dec,('status','reason','candidate_id'))
    values=[s['logit'] for s in scores];order=sorted(range(len(ids)),key=lambda i:(-values[i],i))
    reason=None
    if len(order)>1:
        gap=values[order[0]]-values[order[1]]
        if gap==0:reason='tie'
        elif gap<q['policy']['minimum_logit_margin']:reason='policy_margin'
    check(dec==dict(status='abstained' if reason else 'ranked',reason=reason,candidate_id=None if reason else ids[order[0]]))
    e=d['evidence'];keys(e,('input_sha256','model_revision','tokenizer_revision','artifact_id','ir_id','library_sha256','parity'))
    check(e['input_sha256']==q['input']['sha256'] and e['model_revision']==q['model']['revision'])
    for name in ('artifact_id','ir_id'):check(type(e[name]) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',e[name]))
    for name in ('tokenizer_revision','library_sha256'):
        check(e[name] is None or type(e[name]) is str and re.fullmatch(r'[a-f0-9]{40,64}',e[name]))
    p=e['parity'];keys(p,('status','scope','atol','rtol'))
    check(p['status'] in ('PASS','FAIL','NOT_TESTED') and p['scope'] in ('head_only','end_to_end','none'))
    if p['status']=='NOT_TESTED':check(p['atol'] is None and p['rtol'] is None)
    else:check(all(finite(p[k]) and p[k]>=0 for k in ('atol','rtol')) and p['scope']!='none')
    if d['execution_kind']=='fixture_replay':check(p['status']=='NOT_TESTED' and p['scope']=='none' and e['library_sha256'] is None)
    else:check(type(e['library_sha256']) is str and len(e['library_sha256'])==64)
    if p['scope']=='end_to_end':check(d['implementation_stage']=='end_to_end')
    return copy.deepcopy(d)

class Consumer:
    """Only the latest uncancelled, validated attempt can become current."""
    def __init__(self):self._lock=threading.Lock();self._generation=0;self.current=None
    def cancel(self):
        with self._lock:self._generation+=1;self.current=None
    def evaluate(self,q,exchange,allow_simulation=False):
        q=copy.deepcopy(q)
        with self._lock:self._generation+=1;generation=self._generation;self.current=None
        validate_expected(q)
        deadline=time.monotonic()+q['timeout_ms']/1000
        try:
            reply=exchange(q)
            check(time.monotonic()<deadline,'DEADLINE_EXCEEDED')
            result=validate(reply,q,allow_simulation)
        except InvalidResult:raise
        except Exception:raise InvalidResult('TRANSPORT_ERROR') from None
        with self._lock:
            check(generation==self._generation,'STALE_RESULT')
            # Validation and lock acquisition consume the same attempt budget.
            check(time.monotonic()<deadline,'DEADLINE_EXCEEDED')
            self.current=result
        return copy.deepcopy(result)

class LoopbackHTTP:
    """One bounded connection per attempt; no redirects or remote hosts."""
    def __init__(self,url):
        u=urlsplit(url)
        check(u.scheme=='http' and u.hostname in ('127.0.0.1','::1') and not u.username and not u.password and not u.query and not u.fragment and u.path=='/evaluate','INVALID_ENDPOINT')
        self.host=u.hostname;self.port=u.port or 80
    def __call__(self,q):
        body=json.dumps(q,allow_nan=False).encode();check(len(body)<=MAX_BYTES,'MESSAGE_TOO_LARGE')
        conn=http.client.HTTPConnection(self.host,self.port,timeout=q['timeout_ms']/1000)
        timer=None
        started=time.monotonic()
        try:
            conn.connect()
            sock=conn.sock
            def expire():
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            remaining=max(0.,q['timeout_ms']/1000-(time.monotonic()-started))
            timer=threading.Timer(remaining,expire);timer.daemon=True;timer.start()
            conn.request('POST','/evaluate',body,{'Content-Type':'application/json'})
            response=conn.getresponse();check(response.status in (200,400,409,429,500,504),'HTTP_ERROR')
            check(response.getheader('Content-Type')=='application/json')
            raw=response.read(MAX_BYTES+1);check(len(raw)<=MAX_BYTES,'MESSAGE_TOO_LARGE')
            def pairs(items):
                obj={}
                for k,v in items:check(k not in obj);obj[k]=v
                return obj
            def constant(_):raise InvalidResult('INVALID_RESPONSE')
            return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
        finally:
            if timer is not None:timer.cancel()
            conn.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--allow-simulation',action='store_true');args=parser.parse_args()
    try:
        raw=args.request.read_bytes();check(len(raw)<=MAX_BYTES,'MESSAGE_TOO_LARGE')
        result=Consumer().evaluate(json.loads(raw),LoopbackHTTP(args.url),args.allow_simulation)
        print(json.dumps(dict(status='PASS',result=result),allow_nan=False));return 0
    except (InvalidResult,ValueError,OSError):
        print(json.dumps(dict(status='FAIL',error='EVALUATION_FAILED')));return 1
if __name__=='__main__':raise SystemExit(main())
