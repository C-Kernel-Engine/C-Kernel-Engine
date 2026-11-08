"""Explicit local acceptance: one durable SVG job through the existing workflow."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from server.training.jobs import Jobs,atomic_json
from server.training.job_consumer import observe,verify_artifacts


def run(store,python):
    jobs=Jobs(store,principals={'local-acceptance'},python=python)
    try:
        caps=jobs.backend.capabilities();r=caps['recipes']['svg_tiny_v8'];d=next(d for d in caps['datasets'] if d['id']==r['dataset'])
        config=dict(recipe='svg_tiny_v8',dataset=dict(id=d['id'],version=d['version'],sha256=d['sha256'],train_split='train',validation_split='validation'),
                    shape=r['shape'],tokenizer=r['tokenizer'],dtype=r['dtype'],optimizer=r['optimizer'],method=r['method'],epochs=2,grad_accum=2)
        plan=jobs.prepare('local-acceptance',config,'svg-acceptance',1)
        first=jobs.submit('local-acceptance',plan['plan_sha256'],'svg-acceptance-v1');job_id=first['job']['job_id']
        expected=dict(worker_boot_id=jobs.backend.boot,owner='local-acceptance',job_id=job_id,plan_sha256=plan['plan_sha256'],experiment_id='svg-acceptance',draft_revision=1)
        last=observe(first,**expected)
        while last['job']['status'] in ('queued','running'):
            time.sleep(.5);last=observe(jobs.observe('local-acceptance',job_id),last=last,**expected)
        duplicate=jobs.submit('local-acceptance',plan['plan_sha256'],'svg-acceptance-v1')
        if duplicate['job']['job_id']!=job_id:raise RuntimeError('idempotency failed')
        verified=verify_artifacts(last,lambda artifact_id,offset:jobs.artifact('local-acceptance',job_id,artifact_id,offset))
        result=dict(artifact_roundtrip_verified=verified,schema='cke.training.job-acceptance.v1',evidence='real_generated_c_training',plan_sha256=plan['plan_sha256'],job=last['job'],duplicate_same_job=True)
        atomic_json(Path(store)/'acceptance.json',result)
        return result
    finally:jobs.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--store',type=Path,required=True);p.add_argument('--python',default=sys.executable);a=p.parse_args()
    r=run(a.store,a.python);print(json.dumps(r,indent=2));raise SystemExit(0 if r['job']['status']=='succeeded' else 1)
