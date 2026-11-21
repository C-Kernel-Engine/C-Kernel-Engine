"""Independent application client against registered native LoRA jobs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from server.training.jobs import Jobs, JobError, exchange
from server.training.bridge_consumer import validate
from server.training.job_consumer import verify_artifacts
from version.v8.training.lora_package import load_package, sha
from tests.test_v8_lora_session import base_artifact

ROOT = Path(__file__).resolve().parents[1]


def package(folder):
    folder.mkdir()
    base_artifact(folder / 'base')
    doc = dict(schema='cke.lora.training-package.v1', tokens=4,
               base=dict(directory='base', manifest_sha256=sha(folder/'base/weights_manifest.json'),
                         weights_sha256=sha(folder/'base/weights.bump'), provenance='synthetic'),
               tokenizer=dict(id='synthetic-token-ids', sha256='1'*64),
               records=[dict(id='train', split='train', tokens=[0,1,2,3], targets=[1,2,3,4]),
                        dict(id='validation', split='validation', tokens=[0,1,2,5], targets=[1,2,5,4])])
    path = folder/'package.json';path.write_text(json.dumps(doc))
    return path


class QueuedJobs(Jobs):
    def submit(self, *args, **kwargs):
        return super().submit(*args, **dict(kwargs, start=False))


class LoRAJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.path = package(self.folder/'data')
        self.registry = {'demo': dict(path=str(self.path), sha256=sha(self.path))}
        self.jobs = QueuedJobs(self.folder/'jobs', principals={'a','b'}, lora_packages=self.registry)
        self.addCleanup(self.jobs.close)
        self.revision = 'a'*32
        self.caps = self.call('discover')
        self.config = dict(recipe='decoder_lora_fp32_v1', package_id='demo', package_sha256=sha(self.path),
                           dtype='fp32', optimizer='adamw', method='lora', steps=3, learning_rate=.0003,
                           adapters=[dict(target='layer.0.wq',rank=2,alpha=4,name='query',seed=41),
                                     dict(target='layer.1.wv',rank=3,alpha=6,name='value',seed=42)])

    def call(self, operation, owner='a', **fields):
        q = dict(schema='cke.training.jobs.v2', request_id='client', operation=operation)
        if operation != 'discover':
            q.update(worker_boot_id=self.caps['worker_boot_id'], capabilities_sha256=self.caps['capabilities_sha256'], client_revision=self.revision)
        q.update(fields)
        return validate(exchange(self.jobs,owner,q),q,capabilities=getattr(self,'caps',None),owner=owner)

    def prepare(self):
        return self.call('prepare',config=self.config,experiment_id='lora')

    def test_discovery_prepare_rejections_and_ownership(self):
        self.assertIn('decoder_lora_fp32_v1',self.caps['executable_recipes'])
        self.assertEqual(self.caps['validation']['methods']['lora'],'validation_only')
        with patch('subprocess.Popen',side_effect=AssertionError('prepare cannot launch')), patch('version.v8.training.lora_session.prepare_lora',side_effect=AssertionError('prepare cannot initialize')):
            plan = self.prepare()
        self.assertFalse(plan['submitted'])
        from server.training.consumer import validate as validate_contract
        q=dict(schema='cke.training.v1',request_id='independent',operation='discover')
        response=self.jobs.backend.exchange(q)
        validate_contract(response,q)
        q.update(operation='validate',worker_boot_id=response['worker_boot_id'],capabilities_sha256=response['capabilities_sha256'],config=self.config)
        validate_contract(self.jobs.backend.exchange(q),q,response['capabilities'])
        stale=dict(schema='cke.training.jobs.v2',request_id='stale',operation='submit',worker_boot_id='0'*32,capabilities_sha256=self.caps['capabilities_sha256'],client_revision=self.revision,plan_sha256=plan['plan_sha256'],idempotency_key='stale')
        with self.assertRaisesRegex(JobError,'STALE_WORKER'):exchange(self.jobs,'a',stale)

        for field,value in [('dtype','bf16'),('steps',True),('steps',100001),('package_sha256','0'*64),('path','/tmp/injected')]:
            cfg=dict(self.config,**{field:value})
            with self.subTest(field=field),self.assertRaisesRegex(JobError,'INVALID_CONFIGURATION'):
                self.call('prepare',config=cfg,experiment_id='lora')
        for change in [dict(rank=0),dict(target='missing'),dict(alpha=float('inf')),dict(seed=-1),dict(name='invalid.name')]:
            cfg=copy.deepcopy(self.config);cfg['adapters'][0].update(change)
            with self.subTest(change=change),self.assertRaises((JobError,ValueError)):
                self.call('prepare',config=cfg,experiment_id='lora')
        with self.assertRaisesRegex(JobError,'UNKNOWN_PLAN'):
            self.call('submit',owner='b',plan_sha256=plan['plan_sha256'],idempotency_key='once')
        first=self.call('submit',plan_sha256=plan['plan_sha256'],idempotency_key='once')
        again=self.call('submit',plan_sha256=plan['plan_sha256'],idempotency_key='once')
        self.assertEqual(first['job']['job_id'],again['job']['job_id'])
        with self.assertRaisesRegex(JobError,'UNKNOWN_JOB'):
            self.call('observe',owner='b',job_id=first['job']['job_id'])

    def test_package_integrity_and_split_rejection(self):
        doc=json.loads(self.path.read_text())
        for mutate in ('overlap','invalid_id','duplicate','escape','missing_validation'):
            bad=copy.deepcopy(doc)
            if mutate=='overlap':bad['records'][1]['tokens']=bad['records'][0]['tokens']
            elif mutate=='invalid_id':bad['records'][0]['tokens'][0]=32
            elif mutate=='duplicate':bad['records'][1]['id']='train'
            elif mutate=='escape':bad['base']['directory']='../outside'
            else:bad['records'][1]['split']='train'
            self.path.write_text(json.dumps(bad))
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):load_package(self.path,sha(self.path))
        self.path.write_text(json.dumps(doc))
        plan=self.prepare()
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'identity'):
            self.call('submit',plan_sha256=plan['plan_sha256'],idempotency_key='changed')

    def test_native_job_and_downloads(self):
        if not (ROOT/'build/libckernel_engine.so').exists():self.skipTest('native engine required')
        plan=self.prepare()
        observation=self.call('submit',plan_sha256=plan['plan_sha256'],idempotency_key='native')
        jid=observation['job']['job_id']
        # Invoke the real child executor while developing uncommitted changes.
        # The production worker additionally requires a clean source checkout.
        def execute(command,folder):
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return result.returncode
        with patch.object(self.jobs,'_execute',side_effect=execute):self.jobs.run_job('a',jid)
        done=self.call('observe',job_id=jid)
        self.assertEqual(done['job']['status'],'succeeded',done)
        self.assertEqual(done['job']['progress']['completed_steps'],3)
        self.assertEqual(len(done['job']['metrics']),2)
        types={a['type'] for a in done['job']['artifacts']}
        self.assertTrue({'adapter_package','training_checkpoint','training_report'}<=types)
        verified=verify_artifacts(done,lambda aid,offset:self.call('artifact',job_id=jid,artifact_id=aid,offset=offset))
        self.assertEqual(len(verified),len(done['job']['artifacts']))
        report=json.loads((self.folder/'jobs'/jid/'report.json').read_text())
        self.assertEqual(report['verdicts']['base_provenance'],'synthetic')
        self.assertEqual(report['verdicts']['numerical_oracle'],'NOT_TESTED')
        from server.training.lora import validate_result
        package_doc=json.loads(self.path.read_text())
        for mutation in ('missing_metrics','wrong_mean','missing_record','wrong_config','wrong_verdict'):
            bad=copy.deepcopy(report)
            if mutation=='missing_metrics':bad.pop('metrics')
            elif mutation=='wrong_mean':bad['metrics']['validation_loss_after']+=1
            elif mutation=='missing_record':bad['validation']['after']['records']=[]
            elif mutation=='wrong_config':bad['config_sha256']='0'*64
            else:bad['verdicts']['numerical_oracle']='PASS'
            with self.subTest(mutation=mutation):self.assertFalse(validate_result(bad,self.config,package_doc))

        artifact=next(a for a in done['job']['artifacts'] if a['type']=='adapter_package')
        with open(self.folder/'jobs'/jid/'run/adapters.zip','ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(JobError,'ARTIFACT_CHANGED'):
            self.call('artifact',job_id=jid,artifact_id=artifact['artifact_id'],offset=0)


if __name__=='__main__':unittest.main()
