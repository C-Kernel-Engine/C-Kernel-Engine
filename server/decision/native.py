"""Trusted, registered head packages. Linux native execution; no model scheduler.

The administrator pins package.json's SHA-256. This is executable code, not an
untrusted model upload interface. Inputs are registered supplied features, not
text. Neither NumPy nor PyTorch is imported in the deployed process.
"""
import ctypes
import _ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
import uuid

from . import contract as wire

INPUTS=('hidden','input_ids','question_spans','option_spans','global_span','option_counts','question_type_ids')
MAX_PACKAGE_BYTES=1024**3

def sha(data):return hashlib.sha256(data).hexdigest()

def check(ok):
    if not ok:raise ValueError('INVALID_PACKAGE')

def integer(value, low, high):
    check(type(value) is int and low<=value<=high)
    return value

class NativeHead:
    """One loaded library/weight arena, one caller at a time; no native preemption."""
    def __init__(self, manifest_path, expected_sha256):
        self.fd=None;self.library=None;self.execute=None;self.closed=False
        try:self._load(Path(manifest_path),expected_sha256)
        except Exception:
            self.close()
            raise ValueError('INVALID_PACKAGE') from None
        self.adapter=uuid.uuid4().hex;self.boot=uuid.uuid4().hex
        self.slot=threading.BoundedSemaphore(1)

    def _load(self,path,expected):
        check(sys.byteorder=='little')
        check(wire.matches(wire.HASH,expected) and path.stat().st_size<=4*1024**2)
        raw=path.read_bytes();check(sha(raw)==expected)
        m=json.loads(raw);check(m['schema']=='cke.decision.head-package.v1')
        self.manifest=m;self.package_sha256=expected
        check(wire.matches(wire.IDENT,m['model']['id']) and wire.matches(wire.REV,m['model']['revision']))
        check(wire.matches(wire.REV,m['cke_revision']))
        check(m['tokenizer_revision'] is None)
        def read(record):
            name=record['file'];check(type(name) is str and Path(name).name==name and name not in ('.','..'))
            p=path.parent/name;check(p.stat().st_size<=MAX_PACKAGE_BYTES)
            data=p.read_bytes();check(sha(data)==record['sha256']);return data
        check(set(m['files'])=={'library','weights','layout','ir','circuit','build_identity','generated_source'})
        blobs={key:read(record) for key,record in m['files'].items()}
        build=json.loads(blobs['build_identity'])
        check(build['library_sha256']==sha(blobs['library']))
        check(build['generated_source_sha256']==sha(blobs['generated_source']))
        layout=json.loads(blobs['layout']);graph=json.loads(blobs['circuit'])
        n=integer(m['arena_bytes'],1,MAX_PACKAGE_BYTES)
        check(n==layout['memory']['arena']['total_size'])
        check(m['function']==graph['native_entry']['function'] and wire.matches(wire.IDENT,m['function']))
        acts={e['name']:e for e in layout['memory']['activations']['buffers']}
        planned={e['name']:e for e in layout['memory']['weights']['entries']}
        intervals=[]
        def region(offset,size):
            integer(offset,0,n);integer(size,1,n);check(offset%4==0 and offset+size<=n)
            check(all(offset+size<=a or b<=offset for a,b in intervals));intervals.append((offset,offset+size))
        check(len(m['weight_spans'])==len(planned))
        names=set()
        for w in m['weight_spans']:
            check(w['name'] in planned and w['name'] not in names);names.add(w['name'])
            entry=planned[w['name']];check(w['arena_offset']==entry['abs_offset'])
            check(entry['dtype']=='fp32' and w['size']==entry['size'])
            integer(w['file_offset'],0,len(blobs['weights']));check(w['file_offset']+w['size']<=len(blobs['weights']))
            region(w['arena_offset'],w['size'])
        check(set(m['inputs'])==set(INPUTS))
        for name,s in m['inputs'].items():
            check(s['offset']==acts[name]['abs_offset'])
            check(s['shape']==graph['activation_buffers'][name]['shape'])
            check(s['dtype']==('fp32' if name=='hidden' else 'i32'))
            for dim in s['shape']:integer(dim,1,n//4)
            region(s['offset'],math.prod(s['shape'])*4)
        out=m['output'];check(out['offset']==acts['option_logits']['abs_offset'])
        check(out['count']==math.prod(graph['activation_buffers']['option_logits']['shape']))
        region(out['offset'],out['count']*4)
        self.records={};self.features={}
        tokens=m['inputs']['hidden']['shape'][0]
        check(m['inputs']['input_ids']['shape']==[tokens])
        integer(m['vocabulary'],1,2**31-1)
        check(1<=len(m['artifacts'])<=64)
        for rec in m['artifacts']:
            check(wire.matches(wire.IDENT,rec['id']) and rec['id'] not in self.records)
            check(rec['input_sha256']==wire.digest({k:v for k,v in rec.items() if k!='input_sha256'}))
            data=read(rec);values={};chunks={};cursor=0
            check(len(rec['spans'])==len(INPUTS))
            for span,name in zip(rec['spans'],INPUTS):
                size=math.prod(m['inputs'][name]['shape'])*4
                check(span==dict(name=name,file_offset=cursor,size=size))
                part=data[cursor:cursor+size];check(len(part)==size);cursor+=size
                chunks[name]=part;values[name]=struct.unpack('<'+('f' if name=='hidden' else 'i')*(size//4),part)
            check(cursor==len(data) and all(math.isfinite(x) for x in values['hidden']))
            check(all(0<=x<m['vocabulary'] for x in values['input_ids']))
            for name in ('question_spans','option_spans'):
                v=values[name];check(len(v)%2==0)
                check(all(0<=a<b<=tokens for a,b in zip(v[::2],v[1::2])))
            check(values['global_span']==(tokens-1,tokens))
            counts=values['option_counts'];check(all(x>0 for x in counts) and sum(counts)==out['count'])
            check(len(counts)*2==len(values['question_spans']) and len(values['question_type_ids'])==len(counts))
            check(all(0<=x<3 for x in values['question_type_ids']))
            qi=integer(rec['question_index'],0,len(counts)-1);ids=rec['candidate_ids']
            check(type(ids) is list and len(ids)==counts[qi] and len(ids)<=wire.MAX_CANDIDATES)
            check(all(wire.matches(wire.IDENT,x) for x in ids) and len(set(ids))==len(ids))
            self.records[rec['id']]=rec;self.features[rec['id']]=(chunks,sum(counts[:qi]))
        # Verify exact library bytes and load the same sealed bytes (no path race).
        self.fd=os.memfd_create('cke-decision-head',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
        with os.fdopen(os.dup(self.fd),'wb') as f:f.write(blobs['library'])
        fcntl.fcntl(self.fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL)
        self.library=ctypes.CDLL(f'/proc/self/fd/{self.fd}')
        self.execute=getattr(self.library,m['function']);self.execute.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t];self.execute.restype=ctypes.c_int
        self.storage=ctypes.create_string_buffer(n+63);self.address=(ctypes.addressof(self.storage)+63)&~63
        self.arena=ctypes.cast(self.address,ctypes.POINTER(ctypes.c_uint8))
        ctypes.memset(self.address,0,n)
        for w in m['weight_spans']:
            data=blobs['weights'][w['file_offset']:w['file_offset']+w['size']]
            ctypes.memmove(self.address+w['arena_offset'],data,len(data))
        cursor=0;self.clear_ranges=[]
        for a,b in sorted((w['arena_offset'],w['arena_offset']+w['size']) for w in m['weight_spans']):
            if a>cursor:self.clear_ranges.append((cursor,a-cursor))
            cursor=b
        if cursor<n:self.clear_ranges.append((cursor,n-cursor))

    def close(self):
        # A native call cannot be preempted: wait for ownership before unloading.
        slot=getattr(self,'slot',None)
        if slot is not None:slot.acquire()
        try:
            self.closed=True;self.execute=None
            if self.library is not None:
                # Unload before the /proc fd pathname can be reused by dlopen.
                _ctypes.dlclose(self.library._handle);self.library=None
            if self.fd is not None:os.close(self.fd);self.fd=None
        finally:
            if slot is not None:slot.release()

    def request(self,artifact_id):
        r=self.records[artifact_id]
        return dict(schema=wire.SCHEMA,request_id=uuid.uuid4().hex,adapter_boot_id=self.adapter,worker_boot_id=self.boot,model=self.manifest['model'].copy(),task='registered_feature_scores',input=dict(artifact_id=r['id'],sha256=r['input_sha256']),candidate_ids=list(r['candidate_ids']),timeout_ms=1000,policy=dict(minimum_logit_margin=0.))

    def evaluate(self,q):
        wire.validate_request(q);started=time.monotonic();deadline=started+q['timeout_ms']/1000
        base={k:q[k] for k in ('schema','request_id','adapter_boot_id','worker_boot_id','model')}
        base.update(request_sha256=wire.digest(q),ok=False,result=None,error=None)
        def error(code,status=400):return status,dict(base,error={'code':code})
        if q['adapter_boot_id']!=self.adapter or q['worker_boot_id']!=self.boot:return error('RESTARTED',409)
        if q['model']!=self.manifest['model']:return error('UNKNOWN_MODEL')
        rec=self.records.get(q['input']['artifact_id'])
        if rec is None or q['input']['sha256']!=rec['input_sha256']:return error('UNKNOWN_ARTIFACT')
        if set(q['candidate_ids'])!=set(rec['candidate_ids']):return error('CANDIDATE_MISMATCH')
        if not self.slot.acquire(False):return error('BUSY',429)
        try:
            if self.closed:return error('INTERNAL_ERROR',500)
            for offset,size in self.clear_ranges:ctypes.memset(self.address+offset,0,size)
            chunks,start=self.features[rec['id']]
            for name,data in chunks.items():ctypes.memmove(self.address+self.manifest['inputs'][name]['offset'],data,len(data))
            if time.monotonic()>=deadline:return error('DEADLINE_EXCEEDED',504)
            rc=self.execute(self.arena,self.manifest['arena_bytes'])
            if time.monotonic()>=deadline:return error('DEADLINE_EXCEEDED',504)
            if rc:return error('INTERNAL_ERROR',500)
            values=list((ctypes.c_float*len(rec['candidate_ids'])).from_address(self.address+self.manifest['output']['offset']+start*4))
            scores=dict(zip(rec['candidate_ids'],values));values=[scores[x] for x in q['candidate_ids']]
            try:decision=wire.rank(q['candidate_ids'],values,q['policy']['minimum_logit_margin'])
            except wire.ContractError:return error('NONFINITE_SCORE',500)
            evidence=dict(input_sha256=rec['input_sha256'],model_revision=self.manifest['model']['revision'],tokenizer_revision=None,artifact_id=self.package_sha256,ir_id=self.manifest['files']['ir']['sha256'],library_sha256=self.manifest['files']['library']['sha256'],parity=dict(status='NOT_TESTED',scope='head_only',atol=None,rtol=None))
            result=dict(implementation_stage='head_only',execution_kind='native_generated',scores=[dict(candidate_id=k,logit=v) for k,v in zip(q['candidate_ids'],values)],score_semantics='raw_logits',decision=decision,executed=False,latency_ms=(time.monotonic()-started)*1000,evidence=evidence)
            if time.monotonic()>=deadline:return error('DEADLINE_EXCEEDED',504)
            return 200,dict(base,ok=True,result=result)
        finally:self.slot.release()
