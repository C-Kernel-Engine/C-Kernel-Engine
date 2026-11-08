"""Independent simulated consumer of discovery/validation, never a training job."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from server.training.contract import Backend,ROOT,decode,encode
from server.training.consumer import Consumer,InvalidReply,validate

class TrainingContractTest(unittest.TestCase):
    def setUp(self):self.backend=Backend();self.discovery=dict(schema='cke.training.v1',request_id='discover',operation='discover')
    def request(self,backend=None,recipe='svg_tiny_v8'):
        b=backend or self.backend;c=validate(b.exchange(self.discovery),self.discovery)['capabilities'];r=c['recipes'][recipe];d=next(x for x in c['datasets'] if x['id']==r['dataset'])
        return dict(schema='cke.training.v1',request_id='validate',operation='validate',worker_boot_id=c['worker_boot_id'],capabilities_sha256=c['sha256'],config=dict(recipe=recipe,dataset=dict(id=d['id'],version=d['version'],sha256=d['sha256'],train_split='train',validation_split='validation'),tokenizer=r['tokenizer'],shape=copy.deepcopy(r['shape']),dtype=r['dtype'],optimizer=r['optimizer'],method=r['method'],epochs=2,grad_accum=2))
    def test_acceptance_is_not_submission_and_no_process_launch(self):
        with patch('subprocess.Popen',side_effect=AssertionError('no process')),patch('os.system',side_effect=AssertionError('no shell')),patch.object(Path,'write_bytes',side_effect=AssertionError('no writes')),patch.object(Path,'write_text',side_effect=AssertionError('no writes')):
            for recipe in ('svg_tiny_v8','english_tiny_v8'):
                q=self.request(recipe=recipe);r=Consumer().evaluate(q,self.backend.exchange,capabilities=self.backend.capabilities())
                self.assertTrue(r['ok']);self.assertFalse(r['submitted']);self.assertFalse(r['plan']['submittable'])
                self.assertEqual(r['plan']['execution'],'disabled');self.assertIn('ADMISSION_NOT_IMPLEMENTED',[w['code'] for w in r['warnings']])
                self.assertEqual(r['plan_sha256'],self.backend.exchange(q)['plan_sha256'])
    def test_field_errors_for_unsupported_and_invalid_configs(self):
        mutations=[('recipe','v7_svg_runbook'),('dtype','bf16'),('method','lora'),('method','qlora'),('epochs',True),('grad_accum',0),('tokenizer','bpe'),('optimizer','muon')]
        for field,value in mutations:
            with self.subTest(field=field,value=value):
                q=self.request();q['config'][field]=value;r=validate(self.backend.exchange(q),q)
                self.assertFalse(r['ok']);self.assertIsNone(r['plan']);self.assertIn('/config/'+field,[x['field'] for x in r['errors']])
        for field,value in [('schema','v7'),('operation','submit')]:
            q=self.request();q[field]=value;r=self.backend.exchange(q);self.assertFalse(r['ok'])
        for dim in self.request()['config']['shape']:
            q=self.request();q['config']['shape'][dim]=0;self.assertFalse(self.backend.exchange(q)['ok'])
    def test_dataset_identity_splits_paths_and_commands(self):
        for key,value in [('id','missing'),('sha256','0'*64),('version','v2'),('train_split','validation'),('validation_split','train')]:
            q=self.request();q['config']['dataset'][key]=value;r=self.backend.exchange(q);self.assertFalse(r['ok'])
        for key in ('path','command','shell','output_dir'):
            q=self.request();q['config'][key]='../../private';self.assertFalse(self.backend.exchange(q)['ok'])
        q=self.request();q['config']['dataset']['path']='/etc/passwd';self.assertFalse(self.backend.exchange(q)['ok'])
    def test_restart_resources_and_changed_capability(self):
        q=self.request();other=Backend();self.assertIn('STALE_WORKER',[e['code'] for e in other.exchange(q)['errors']])
        b=Backend(resources='unavailable');q=self.request(b);self.assertIn('UNAVAILABLE',[e['code'] for e in b.exchange(q)['errors']])
        q=self.request();q['capabilities_sha256']='0'*64;self.assertIn('STALE_CAPABILITIES',[e['code'] for e in self.backend.exchange(q)['errors']])
    def test_independent_consumer_rejects_missing_and_changed_identity(self):
        q=self.request();good=self.backend.exchange(q)
        for key in ('request_sha256','worker_boot_id','capabilities_sha256','plan_sha256','plan'):
            bad=copy.deepcopy(good);bad[key]=None
            with self.subTest(key=key),self.assertRaises((InvalidReply,TypeError)):validate(bad,q,self.backend.capabilities())
        bad=copy.deepcopy(good);bad['plan']['config']['epochs']=9
        with self.assertRaises(InvalidReply):validate(bad,q,self.backend.capabilities())
        consumer=Consumer();consumer.evaluate(q,self.backend.exchange,capabilities=self.backend.capabilities())
        with self.assertRaises(InvalidReply):consumer.evaluate(q,lambda _:bad,capabilities=self.backend.capabilities())
        self.assertIsNone(consumer.current)
    def test_deadline_validation_delay_and_cancelled_late_reply(self):
        from server.training import consumer as module
        q=self.request();c=Consumer();original=module.validate
        def slow(*args):time.sleep(.03);return original(*args)
        with patch.object(module,'validate',slow),self.assertRaises(InvalidReply):c.evaluate(q,self.backend.exchange,timeout_ms=10,capabilities=self.backend.capabilities())
        self.assertIsNone(c.current)
        entered=threading.Event();release=threading.Event();errors=[]
        def exchange(q):entered.set();release.wait(2);return self.backend.exchange(q)
        def run():
            try:c.evaluate(q,exchange,capabilities=self.backend.capabilities())
            except InvalidReply:errors.append('stale')
        t=threading.Thread(target=run);t.start()
        try:self.assertTrue(entered.wait(1));c.cancel()
        finally:release.set();t.join()
        self.assertEqual(errors,['stale']);self.assertIsNone(c.current)
        self.assertTrue(c.evaluate(q,self.backend.exchange,capabilities=self.backend.capabilities())['ok'])
    def test_changed_registry_and_unavailable_resources_are_not_validated(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.git').write_text((ROOT/'.git').read_text() if (ROOT/'.git').is_file() else 'gitdir: '+str(ROOT/'.git'))
            shutil.copytree(ROOT/'version/v8/training',root/'version/v8/training')
            (root/'version/v8/scripts').mkdir();shutil.copy(ROOT/'version/v8/scripts/run_training_workflow_v8.py',root/'version/v8/scripts')
            b=Backend(root);q=self.request(b)
            path=root/'version/v8/training/corpus/svg_train.svg';path.write_bytes(path.read_bytes()+b'bad')
            result=b.exchange(q);codes=[e['code'] for e in result['errors']]
            self.assertIn('STALE_CAPABILITIES',codes);self.assertIn('QC_FAILED',codes)
            self.assertIsNone(result['plan'])
    def test_snapshot_and_forged_plan_cannot_create_false_pass(self):
        from server.training.consumer import sha
        q=self.request();caps=self.backend.capabilities();good=self.backend.exchange(q)
        bad=copy.deepcopy(good);bad['plan']['cke_commit']='0'*40;bad['plan_sha256']=sha(bad['plan'])
        with self.assertRaises(InvalidReply):validate(bad,q,caps)
        q['config']['method']='qlora';bad=copy.deepcopy(good);bad['request_sha256']=sha(q);bad['plan']['config']=q['config'];bad['plan_sha256']=sha(bad['plan'])
        with self.assertRaises(InvalidReply):validate(bad,q,caps)
        with self.assertRaises(InvalidReply):validate(good,self.request())

    def test_discovery_snapshot_is_isolated_and_requires_runtime_identity(self):
        from server.training.consumer import sha
        caps=self.backend.capabilities();caps['recipes']['svg_tiny_v8']['shape']['d_model']=999
        self.assertEqual(self.backend.capabilities()['recipes']['svg_tiny_v8']['shape']['d_model'],32)
        bad=self.backend.exchange(self.discovery);del bad['capabilities']['runtime']['version']
        c=bad['capabilities'];c['sha256']=sha({k:v for k,v in c.items() if k!='sha256'});bad['capabilities_sha256']=c['sha256']
        with self.assertRaises(InvalidReply):validate(bad,self.discovery)

    def test_versioned_consumer_fixture(self):
        f=json.loads((ROOT/'server/training/fixtures/validation_v1.json').read_text())
        caps=validate(f['discovery_response'],f['discovery_request'])['capabilities']
        result=validate(f['validation_response'],f['validation_request'],caps)
        self.assertFalse(result['submitted']);self.assertEqual(result['plan']['execution'],'disabled')

    def test_bounded_strict_json(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'x'*16385):
            with self.assertRaises(ValueError):decode(raw)
    def test_independent_stdio_process(self):
        p=subprocess.Popen([sys.executable,'-S','-m','server.training.contract'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            p.stdin.write(encode(self.discovery)+b'\n');p.stdin.flush();r=validate(json.loads(p.stdout.readline()),self.discovery);caps=r['capabilities']
            q=self.request();q.update(worker_boot_id=caps['worker_boot_id'],capabilities_sha256=caps['sha256'])
            p.stdin.write(encode(q)+b'\n');p.stdin.flush();answer=validate(json.loads(p.stdout.readline()),q,caps)
            self.assertTrue(answer['ok']);self.assertFalse(answer['submitted'])
        finally:p.terminate();p.communicate(timeout=3)

if __name__=='__main__':unittest.main()
