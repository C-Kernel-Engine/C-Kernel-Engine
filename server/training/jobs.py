"""Opt-in local registered training recipes. No network authentication or remote deployment.

Identity is supplied by the trusted caller, never a request-body field. sqlite
receipts are durable across reconnects; ambiguous interrupted work is not retried.
"""
import base64
import copy
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from .contract import Backend,ROOT,SHAPE,digest,encode,decode,file_hash,ident,MAX_BYTES

from version.v8.decision.training_limits import MAX_STEPS

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
    def __init__(self,store,*,principals,python=sys.executable,root=ROOT,min_free_bytes=2*1024**3,clef_packages=None,wall_seconds=600):
        require(principals and all(ident(p) for p in principals),'INVALID_PRINCIPALS')
        require(type(wall_seconds) is int and 60<=wall_seconds<=86400,'INVALID_WALL_BUDGET')
        self.wall_seconds=wall_seconds
        self.principals=frozenset(principals);self.root=Path(root);self.backend=Backend(self.root,clef_packages=clef_packages)
        self.python=str(Path(python).absolute());self.min_free_bytes=min_free_bytes
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
                    execution='opt_in_registered_native' if self.backend.clef_packages else 'opt_in_local_svg',executable_recipes=['svg_tiny_v8']+(['clef_head_fp32_v1'] if self.backend.clef_packages else []),operations=['discover','prepare','submit','observe','artifact'],cancel='unsupported',
                    limits=dict(active_jobs=1,queued_backlog=0,registered_jobs=32,plan_receipts=128,log_bytes=1048576,artifact_chunk_bytes=8192,wall_seconds=self.wall_seconds),
                    authorization='trusted_local_caller; no network authentication',resources='disk_checked; one local slot; no global reservation')
    def prepare(self,principal,config,experiment_id,draft_revision=None,*,client_revision=None):
        self._auth(principal);require(ident(experiment_id),'INVALID_EXPERIMENT')
        if client_revision is None:require(type(draft_revision) is int and draft_revision>=0,'INVALID_EXPERIMENT')
        else:require(draft_revision is None and type(client_revision) is str and re.fullmatch('[a-f0-9]{32}',client_revision),'INVALID_CLIENT_REVISION')
        require(type(config) is dict,'INVALID_CONFIGURATION')
        clef=config.get('recipe')=='clef_head_fp32_v1' and bool(self.backend.clef_packages)
        require(config.get('recipe')=='svg_tiny_v8' or clef,'UNSUPPORTED_RECIPE')
        if not clef:require(type(config.get('epochs')) is int and 2<=config['epochs']<=10,'EPOCHS_REQUIRE_TWO_OR_MORE')
        # Existing certification requires loss reduction between first/last epochs.
        self._resources();caps=self.backend.capabilities()
        q=dict(schema='cke.training.v1',request_id='prepare',operation='validate',worker_boot_id=self.backend.boot,capabilities_sha256=caps['sha256'],config=config)
        result=self.backend.exchange(q);require(result['ok'],'INVALID_CONFIGURATION')
        plan=dict(schema='cke.training.execution-plan.v1',owner=principal,experiment_id=experiment_id,draft_revision=draft_revision,
                  worker_boot_id=self.backend.boot,validation_plan=result['plan'],validation_plan_sha256=result['plan_sha256'],
                  executor='v8_generated_training_workflow',python_sha256=file_hash(Path(self.python)),seed=42,
                  optimizer=dict(lr=config['learning_rate'] if clef else .0003,beta1=.9,beta2=.999,eps=1e-4 if clef else 1e-8,weight_decay=.01),
                  submittable=True,expires_at=int(time.time())+600)
        if client_revision is not None:plan.update(schema='cke.training.execution-plan.v2',client_revision=client_revision)
        h=digest(plan)
        with self.guard,self.db:
            require(self.db.execute('SELECT count(*) FROM plans').fetchone()[0]<128,'PLAN_LIMIT')
            self.db.execute('INSERT OR IGNORE INTO plans VALUES(?,?,?)',(h,principal,encode(plan).decode()))
        return dict(plan=plan,plan_sha256=h,submitted=False)
    def submit(self,principal,plan_sha256,idempotency_key,*,start=True,client_revision=None):
        self._auth(principal);require(ident(idempotency_key),'INVALID_IDEMPOTENCY_KEY')
        with self.guard,self.db:
            row=self.db.execute('SELECT plan_hash,body FROM jobs WHERE owner=? AND idem=?',(principal,idempotency_key)).fetchone()
            if row:
                require(row[0]==plan_sha256,'IDEMPOTENCY_CONFLICT');prior=json.loads(row[1])
                require(prior.get('client_revision')==client_revision,'CLIENT_REVISION_MISMATCH');return self._observation(prior)
            row=self.db.execute('SELECT body FROM plans WHERE hash=? AND owner=?',(plan_sha256,principal)).fetchone()
            require(row is not None,'UNKNOWN_PLAN');plan=json.loads(row[0]);require(digest(plan)==plan_sha256,'PLAN_CHANGED');require(plan.get('client_revision')==client_revision,'CLIENT_REVISION_MISMATCH');self._recheck(plan)
            require(self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]<32,'JOB_LIMIT')
            active=any(json.loads(r[0])['status'] in ('queued','running','recovery_required') for r in self.db.execute('SELECT body FROM jobs'))
            require(not active,'BUSY');self._resources()
            job_id=uuid.uuid4().hex;now=time.time()
            j=dict(schema='cke.training.job.v1',job_id=job_id,owner=principal,experiment_id=plan['experiment_id'],draft_revision=plan['draft_revision'],
                   submitted_worker_boot_id=self.backend.boot,cke_commit=plan['validation_plan']['cke_commit'],plan_sha256=plan_sha256,
                   status='queued',stage='admission',created_at=now,updated_at=now,error_code=None,artifacts=[],metrics=[],
                   execution_kind=self.execution_kind,progress=None)
            if client_revision is not None:j.update(schema='cke.training.job.v2',client_revision=client_revision)
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
        with self.guard,self.db:
            j=self._read(principal,job_id)
            progress=self.store/job_id/'run/progress.json'
            if j['status']=='running' and progress.is_file() and not progress.is_symlink() and progress.stat().st_size<=16384:
                try:
                    row=json.loads(progress.read_text())
                    valid=(type(row) is dict and row.get('stage') in ('training','complete') and type(row.get('completed_steps')) is int and type(row.get('total_steps')) is int and 0<=row['completed_steps']<=row['total_steps']<=MAX_STEPS and (row['stage']!='complete' or row['completed_steps']==row['total_steps']))
                    if valid:
                        value={k:row[k] for k in ('stage','completed_steps','total_steps')}
                        prior=j['progress']
                        monotonic=(prior is None or (value['total_steps']==prior['total_steps'] and value['completed_steps']>=prior['completed_steps'] and (prior['stage']!='complete' or value['stage']=='complete')))
                        if monotonic and value!=prior:
                            j.update(progress=value,updated_at=time.time());self._save(j)
                except (ValueError,OSError):pass
            return self._observation(j)
    def _command(self,plan,folder):
        c=plan['validation_plan']['config']
        if c['recipe']=='clef_head_fp32_v1':
            package=self.backend.clef_packages[c['package_id']]
            return [self.python,str(self.root/'version/v8/decision/train_clef_head.py'),'--package',str(Path(package['path']).resolve()),'--package-sha256',c['package_sha256'],'--run-dir',str(folder/'run'),'--json-out',str(folder/'report.json'),'--epochs',str(c['epochs']),'--lr',str(c['learning_rate'])]
        s=c['shape']
        args=[self.python,str(self.root/'version/v8/scripts/run_training_workflow_v8.py'),
              '--run-dir',str(folder/'run'),'--json-out',str(folder/'report.json'),
              '--corpus',str(self.root/'version/v8/training/svg_single_document_v1.json'),'--tokenizer','byte','--seed',str(plan['seed'])]
        for flag,value in [('layers',s['layers']),('d-model',s['d_model']),('hidden',s['hidden']),('num-heads',s['heads']),('num-kv-heads',s['kv_heads']),('vocab-size',s['vocab_size']),('seq-len',s['seq_len']),('epochs',c['epochs']),('grad-accum',c['grad_accum']),*plan['optimizer'].items()]:
            args.extend(['--'+flag.replace('_','-'),str(value)])
        return args
    def _execute(self,command,folder):
        env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
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
        timer=threading.Timer(self.wall_seconds,expire);timer.daemon=True;timer.start()
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
            clef=plan['validation_plan']['config']['recipe']=='clef_head_fp32_v1'
            if clef:required={'native_execution','frozen_embedding','checkpoint_reload','inference_export'}
            checks=report.get('checks',{})
            passed=(rc==0 and report.get('passed') is True and report.get('status')=='PASS' and required<=checks.keys() and all(type(v) is dict and v.get('passed') is True for v in checks.values()))
            artifacts=[]
            files=[('validated_plan.json','execution_plan'),('report.json','training_report')]
            for name,kind in [('run/training_plan.json','training_plan'),('run/training_pipeline_latest.json','training_pipeline')]:
                if (folder/name).is_file():files.append((name,kind))
            checkpoint=report.get('artifacts',{}).get('checkpoint')
            if checkpoint:
                path=Path(checkpoint)/'checkpoint.json'
                require(path.resolve().is_relative_to(folder.resolve()) and path.is_file(),'INVALID_CHECKPOINT_ARTIFACT')
                files.append((str(path.relative_to(folder)),'checkpoint_manifest'))
            if clef and report.get('passed') is True:
                require(report.get('schema')=='cke.clef.training-result.v1' and report.get('package_sha256')==plan['validation_plan']['config']['package_sha256'],'RESULT_IDENTITY_MISMATCH')
                files.extend([('run/trained_head.npz','trained_head'),('run/serving.zip','inference_package')])
            for name,kind in files:
                path=folder/name
                require(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()) and 0<path.stat().st_size<=(512 if clef else 16)*1024**2,'INVALID_ARTIFACT')
                artifacts.append(dict(artifact_id=uuid.uuid4().hex,type=kind,bytes=path.stat().st_size,sha256=file_hash(path),file=name))
            metrics=[];observed=time.time()
            for source,name in [('last_epoch_loss','training_cross_entropy'),('heldout_loss_after','validation_cross_entropy')]:
                value=checks.get('learning',{}).get(source)
                if type(value) in (float,int) and math.isfinite(value):
                    metrics.append(dict(name=name,step=plan['validation_plan']['config']['epochs'],step_kind='epoch',value=value,unit='nats/token',observed_at=observed))
            if clef:
                for source,name in [('validation_loss_before','validation_cross_entropy_before'),('validation_loss_after','validation_cross_entropy')]:
                    value=report.get('metrics',{}).get(source)
                    if type(value) in (float,int) and math.isfinite(value):metrics.append(dict(name=name,step=report['completed_steps'],step_kind='update',value=value,unit='nats/question',observed_at=observed))
            with self.guard,self.db:
                j=self._read(principal,job_id)
                if clef and passed:j['progress']=dict(stage='complete',completed_steps=report['completed_steps'],total_steps=report['completed_steps'])
                j.update(status='succeeded'  if passed else 'failed',stage='complete',updated_at=time.time(),error_code=None if passed else 'WORKFLOW_CERTIFICATION_FAILED',artifacts=artifacts,metrics=metrics);self._save(j)
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
    if type(q) is dict and q.get('schema')=='cke.training.jobs.v2':
        from .jobs_v2 import exchange as exchange_v2
        return exchange_v2(jobs,principal,q)
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
    p.add_argument('--enable-svg-training',action='store_true')
    p.add_argument('--clef-registry',type=Path,help='Administrator-owned JSON mapping IDs to pinned package paths/hashes')
    p.add_argument('--store',type=Path,required=True);p.add_argument('--principal',required=True);p.add_argument('--python',default=sys.executable)
    a=p.parse_args()
    if not a.enable_svg_training and not a.clef_registry:p.error('explicit local training opt-in required')
    jobs=Jobs(a.store,principals={a.principal},python=a.python,clef_packages=json.loads(a.clef_registry.read_text()) if a.clef_registry else None)
    try:
        while True:
            raw=sys.stdin.buffer.readline(MAX_BYTES+1)
            if not raw:return
            q=None
            try:
                q=decode(raw);reply=exchange(jobs,a.principal,q)
            except Exception as exc:
                reply=dict(schema='cke.training.jobs.v2' if type(q) is dict and q.get('schema')=='cke.training.jobs.v2' else 'cke.training.jobs.v1',request_id=q.get('request_id') if type(q) is dict and ident(q.get('request_id')) else None,ok=False,result=None,error=str(exc) if isinstance(exc,JobError) else 'INVALID_REQUEST')
            body=encode(reply)
            if len(body)>MAX_BYTES:body=encode(dict(schema=reply['schema'],ok=False,error='RESPONSE_TOO_LARGE'))
            print(body.decode(),flush=True)
            if len(raw)>MAX_BYTES:return
    finally:jobs.close()

if __name__=='__main__':main()
