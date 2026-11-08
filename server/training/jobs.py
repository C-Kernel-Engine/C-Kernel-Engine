"""Opt-in local SVG job adapter. No network authentication or remote deployment.

Identity is supplied by the trusted caller, never a request-body field. sqlite
receipts are durable across reconnects; ambiguous interrupted work is not retried.
"""
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from .contract import Backend,ROOT,SHAPE,digest,encode,decode,file_hash,ident,MAX_BYTES

class JobError(ValueError):pass

def require(ok,code):
    if not ok:raise JobError(code)

def atomic_json(path,data):
    temp=path.with_suffix('.tmp');raw=encode(data)
    with open(temp,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

class Jobs:
    execution_kind="native_generated_workflow"
    """Single admitted job, no backlog. Store is private to one local daemon."""
    def __init__(self,store,*,principals,python=sys.executable,root=ROOT,min_free_bytes=2*1024**3):
        require(principals and all(ident(p) for p in principals),'INVALID_PRINCIPALS')
        self.principals=frozenset(principals);self.root=Path(root);self.backend=Backend(self.root)
        self.python=str(Path(python).resolve());self.min_free_bytes=min_free_bytes
        self.store=Path(store).resolve();self.store.mkdir(parents=True,exist_ok=True,mode=0o700)
        require(self.store.stat().st_uid==os.getuid() and self.store.stat().st_mode&0o077==0,'PRIVATE_STORE_REQUIRED')
        self.lockfile=open(self.store/'worker.lock','a+')
        try:fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:self.lockfile.close();raise JobError('STORE_BUSY') from None
        self.guard=threading.RLock();self.db=sqlite3.connect(self.store/'jobs.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS plans(hash TEXT PRIMARY KEY,owner TEXT NOT NULL,body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,owner TEXT NOT NULL,idem TEXT NOT NULL,plan_hash TEXT NOT NULL,body TEXT NOT NULL,UNIQUE(owner,idem));''')
        self.threads=[]
        # Never re-launch a job whose launch/completion receipt may have been lost.
        with self.db:
            for job_id,body in self.db.execute('SELECT id,body FROM jobs').fetchall():
                j=json.loads(body)
                if j['status'] in ('queued','running'):
                    j.update(status='recovery_required',stage='unknown',error_code='WORKER_INTERRUPTED',updated_at=time.time());self._save(j)
    def close(self):
        for t in self.threads:t.join()
        self.db.close();self.lockfile.close()
    def _auth(self,principal):require(principal in self.principals,'ACCESS_DENIED')
    def _save(self,j):self.db.execute('UPDATE jobs SET body=? WHERE id=?',(encode(j).decode(),j['job_id']))
    def _read(self,principal,job_id):
        self._auth(principal);require(ident(job_id),'UNKNOWN_JOB')
        row=self.db.execute('SELECT body FROM jobs WHERE id=? AND owner=?',(job_id,principal)).fetchone()
        require(row is not None,'UNKNOWN_JOB');return json.loads(row[0])
    def _resources(self):
        v=os.statvfs(self.store);require(v.f_bavail*v.f_frsize>=self.min_free_bytes,'RESOURCES_UNAVAILABLE')
    def capabilities(self,principal):
        self._auth(principal)
        return dict(schema='cke.training.jobs.v1',worker_boot_id=self.backend.boot,validation=self.backend.capabilities(),
                    execution='opt_in_local_svg',operations=['prepare','submit','observe','artifact'],cancel='unsupported',
                    limits=dict(active_jobs=1,queued_backlog=0,registered_jobs=32,plan_receipts=128,log_bytes=1048576,artifact_chunk_bytes=8192,wall_seconds=600),
                    authorization='trusted_local_caller; no network authentication',resources='disk_checked; one local slot; no global reservation')
    def prepare(self,principal,config,experiment_id,draft_revision):
        self._auth(principal);require(ident(experiment_id) and type(draft_revision) is int and draft_revision>=0,'INVALID_EXPERIMENT')
        require(type(config) is dict and config.get('recipe')=='svg_tiny_v8','UNSUPPORTED_RECIPE')
        require(type(config.get('epochs')) is int and 2<=config['epochs']<=10,'EPOCHS_REQUIRE_TWO_OR_MORE')
        # Existing certification requires loss reduction between first/last epochs.
        self._resources();caps=self.backend.capabilities()
        q=dict(schema='cke.training.v1',request_id='prepare',operation='validate',worker_boot_id=self.backend.boot,capabilities_sha256=caps['sha256'],config=config)
        result=self.backend.exchange(q);require(result['ok'],'INVALID_CONFIGURATION')
        plan=dict(schema='cke.training.execution-plan.v1',owner=principal,experiment_id=experiment_id,draft_revision=draft_revision,
                  worker_boot_id=self.backend.boot,validation_plan=result['plan'],validation_plan_sha256=result['plan_sha256'],
                  executor='v8_generated_training_workflow',python_sha256=file_hash(Path(self.python)),seed=42,
                  optimizer=dict(lr=.0003,beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01),
                  submittable=True,expires_at=int(time.time())+600)
        h=digest(plan)
        with self.guard,self.db:
            require(self.db.execute('SELECT count(*) FROM plans').fetchone()[0]<128,'PLAN_LIMIT')
            self.db.execute('INSERT OR IGNORE INTO plans VALUES(?,?,?)',(h,principal,encode(plan).decode()))
        return dict(plan=plan,plan_sha256=h,submitted=False)
    def submit(self,principal,plan_sha256,idempotency_key,*,start=True):
        self._auth(principal);require(ident(idempotency_key),'INVALID_IDEMPOTENCY_KEY')
        with self.guard,self.db:
            row=self.db.execute('SELECT plan_hash,body FROM jobs WHERE owner=? AND idem=?',(principal,idempotency_key)).fetchone()
            if row:
                require(row[0]==plan_sha256,'IDEMPOTENCY_CONFLICT');return self._observation(json.loads(row[1]))
            row=self.db.execute('SELECT body FROM plans WHERE hash=? AND owner=?',(plan_sha256,principal)).fetchone()
            require(row is not None,'UNKNOWN_PLAN');plan=json.loads(row[0]);require(digest(plan)==plan_sha256,'PLAN_CHANGED');self._recheck(plan)
            require(self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]<32,'JOB_LIMIT')
            active=any(json.loads(r[0])['status'] in ('queued','running','recovery_required') for r in self.db.execute('SELECT body FROM jobs'))
            require(not active,'BUSY');self._resources()
            job_id=uuid.uuid4().hex;now=time.time()
            j=dict(schema='cke.training.job.v1',job_id=job_id,owner=principal,experiment_id=plan['experiment_id'],draft_revision=plan['draft_revision'],
                   submitted_worker_boot_id=self.backend.boot,cke_commit=plan['validation_plan']['cke_commit'],plan_sha256=plan_sha256,
                   status='queued',stage='admission',created_at=now,updated_at=now,error_code=None,artifacts=[],metrics=[],
                   execution_kind=self.execution_kind,progress=None)
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',(job_id,principal,idempotency_key,plan_sha256,encode(j).decode()))
        if start:
            t=threading.Thread(target=self.run_job,args=(principal,job_id));self.threads.append(t);t.start()
        return self._observation(j)
    def _recheck(self,plan):
        require(plan['python_sha256']==file_hash(Path(self.python)),'EXECUTOR_CHANGED')
        require(plan['worker_boot_id']==self.backend.boot,'STALE_WORKER')
        require(time.time()<plan['expires_at'],'PLAN_EXPIRED')
        require(plan['validation_plan_sha256']==digest(plan['validation_plan']),'PLAN_CHANGED')
        require(plan['validation_plan']['capabilities_sha256']==self.backend.capabilities()['sha256'],'CAPABILITIES_CHANGED')
    def _observation(self,j):
        j=copy.deepcopy(j)
        for a in j['artifacts']:a.pop('file',None)
        return dict(job=j,worker_boot_id=self.backend.boot,observed_at=time.time(),cancel_supported=False)
    def observe(self,principal,job_id):
        with self.guard:return self._observation(self._read(principal,job_id))
    def _command(self,plan,folder):
        c=plan['validation_plan']['config'];s=c['shape']
        args=[self.python,str(self.root/'version/v8/scripts/run_training_workflow_v8.py'),
              '--run-dir',str(folder/'run'),'--json-out',str(folder/'report.json'),
              '--corpus',str(self.root/'version/v8/training/svg_single_document_v1.json'),'--tokenizer','byte','--seed',str(plan['seed'])]
        for flag,value in [('layers',s['layers']),('d-model',s['d_model']),('hidden',s['hidden']),('num-heads',s['heads']),('num-kv-heads',s['kv_heads']),('vocab-size',s['vocab_size']),('seq-len',s['seq_len']),('epochs',c['epochs']),('grad-accum',c['grad_accum']),*plan['optimizer'].items()]:
            args.extend(['--'+flag.replace('_','-'),str(value)])
        return args
    def _execute(self,command,folder):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        # Require an unchanged tracked implementation at execution; preparation itself
        # never invokes git, a compiler, subprocess, or training arithmetic.
        clean=subprocess.run(['git','diff','--quiet','HEAD','--','src','include','version/v7','version/v8','server/training'],cwd=self.root)
        require(clean.returncode==0,'SOURCE_DIRTY')
        timed_out=threading.Event()
        proc=subprocess.Popen(command,cwd=self.root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        def expire():
            timed_out.set()
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        timer=threading.Timer(600,expire);timer.daemon=True;timer.start()
        try:
            with open(folder/'workflow.log','wb') as log:
                remaining=1048576
                while True:
                    chunk=proc.stdout.read(8192)
                    if not chunk:break
                    if remaining:log.write(chunk[:remaining]);remaining=max(0,remaining-len(chunk))
            rc=proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            proc.stdout.close()
        require(not timed_out.is_set(),'EXECUTION_TIMEOUT');return rc
    def run_job(self,principal,job_id):
        with self.guard,self.db:
            j=self._read(principal,job_id);require(j['status']=='queued','NOT_QUEUED')
            plan=json.loads(self.db.execute('SELECT body FROM plans WHERE hash=?',(j['plan_sha256'],)).fetchone()[0])
            j.update(status='running',stage='workflow',updated_at=time.time());self._save(j)
        folder=self.store/job_id
        try:
            require(digest(plan)==j['plan_sha256'],'PLAN_CHANGED');self._recheck(plan);self._resources();folder.mkdir(mode=0o700)
            atomic_json(folder/'validated_plan.json',plan)
            command=self._command(plan,folder);atomic_json(folder/'command.json',dict(argv=command))
            rc=self._execute(command,folder)
            require((folder/'report.json').is_file() and (folder/'report.json').stat().st_size<=16*1024**2,'MISSING_OR_OVERSIZED_REPORT')
            report=json.loads((folder/'report.json').read_text())
            required={'learning','pytorch_trajectory','fresh_process_resume','checkpoint_compatibility','inference_export','runtime_provenance','training_ir_visualizer'}
            checks=report.get('checks',{})
            passed=(rc==0 and report.get('passed') is True and report.get('status')=='PASS' and required<=checks.keys() and all(type(v) is dict and v.get('passed') is True for v in checks.values()))
            artifacts=[]
            for name,kind in [('validated_plan.json','execution_plan'),('report.json','training_report')]:
                path=folder/name;artifacts.append(dict(artifact_id=uuid.uuid4().hex,type=kind,bytes=path.stat().st_size,sha256=file_hash(path),file=name))
            with self.guard,self.db:
                j=self._read(principal,job_id);j.update(status='succeeded' if passed else 'failed',stage='complete',updated_at=time.time(),error_code=None if passed else 'WORKFLOW_CERTIFICATION_FAILED',artifacts=artifacts);self._save(j)
        except Exception as exc:
            code=str(exc) if isinstance(exc,JobError) else 'EXECUTION_FAILED'
            with self.guard,self.db:
                j=self._read(principal,job_id);j.update(status='failed',stage='complete',updated_at=time.time(),error_code=code);self._save(j)
    def artifact(self,principal,job_id,artifact_id,offset=0):
        require(type(offset) is int and offset>=0,'INVALID_OFFSET')
        with self.guard:
            j=self._read(principal,job_id);a=next((x for x in j['artifacts'] if x['artifact_id']==artifact_id),None);require(a is not None,'UNKNOWN_ARTIFACT')
            path=self.store/job_id/a['file'];require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.store/job_id),'INVALID_ARTIFACT')
            require(file_hash(path)==a['sha256'],'ARTIFACT_CHANGED');require(offset<=a['bytes'],'INVALID_OFFSET')
            with open(path,'rb') as f:f.seek(offset);data=f.read(8192)
            return dict(artifact_id=artifact_id,sha256=a['sha256'],bytes=a['bytes'],offset=offset,next_offset=offset+len(data),eof=offset+len(data)==a['bytes'],base64=base64.b64encode(data).decode())

def exchange(jobs,principal,q):
    require(type(q) is dict and q.get('schema')=='cke.training.jobs.v1' and ident(q.get('request_id')),'INVALID_REQUEST')
    operation=q.get('operation')
    fields={'discover':set(),'prepare':{'config','experiment_id','draft_revision'},'submit':{'plan_sha256','idempotency_key'},'observe':{'job_id'},'artifact':{'job_id','artifact_id','offset'}}
    require(type(operation) is str and operation in fields,'UNSUPPORTED_OPERATION')
    require(set(q)=={'schema','request_id','operation'}|fields[operation] and len(encode(q))<=MAX_BYTES,'INVALID_REQUEST')
    if operation=='discover':result=jobs.capabilities(principal)
    elif operation=='prepare':result=jobs.prepare(principal,q['config'],q['experiment_id'],q['draft_revision'])
    elif operation=='submit':result=jobs.submit(principal,q['plan_sha256'],q['idempotency_key'])
    elif operation=='observe':result=jobs.observe(principal,q['job_id'])
    else:result=jobs.artifact(principal,q['job_id'],q['artifact_id'],q['offset'])
    return dict(schema='cke.training.jobs.v1',request_id=q['request_id'],request_sha256=digest(q),worker_boot_id=jobs.backend.boot,ok=True,result=result,error=None)

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable-svg-training',action='store_true',required=True)
    p.add_argument('--store',type=Path,required=True);p.add_argument('--principal',required=True);p.add_argument('--python',default=sys.executable)
    a=p.parse_args();jobs=Jobs(a.store,principals={a.principal},python=a.python)
    try:
        while True:
            raw=sys.stdin.buffer.readline(MAX_BYTES+1)
            if not raw:return
            q=None
            try:
                q=decode(raw);reply=exchange(jobs,a.principal,q)
            except Exception as exc:
                reply=dict(schema='cke.training.jobs.v1',request_id=q.get('request_id') if type(q) is dict and ident(q.get('request_id')) else None,ok=False,result=None,error=str(exc) if isinstance(exc,JobError) else 'INVALID_REQUEST')
            body=encode(reply)
            if len(body)>MAX_BYTES:body=encode(dict(schema='cke.training.jobs.v1',ok=False,error='RESPONSE_TOO_LARGE'))
            print(body.decode(),flush=True)
            if len(raw)>MAX_BYTES:return
    finally:jobs.close()

if __name__=='__main__':main()
