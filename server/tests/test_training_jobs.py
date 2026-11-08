"""Job-contract simulations are separate from the opt-in real workflow test."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from server.training.jobs import Jobs,JobError,atomic_json
from server.training.contract import digest
from server.tests import test_training_contract as validation_tests

class SimulatedJobs(Jobs):
    execution_kind='contract_simulation'
    calls=0
    def _execute(self,command,folder):
        self.calls+=1
        checks={k:{'passed':True} for k in ('learning','pytorch_trajectory','fresh_process_resume','checkpoint_compatibility','inference_export','runtime_provenance','training_ir_visualizer')}
        atomic_json(folder/'report.json',dict(status='PASS',passed=True,checks=checks,evidence='contract_simulation'))
        return 0

class TrainingJobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.jobs=SimulatedJobs(Path(self.tmp.name)/'store',principals={'alice','bob'});self.addCleanup(lambda:self.jobs.close())
        helper=validation_tests.TrainingContractTest();helper.setUp();self.config=helper.request(self.jobs.backend)['config']
    def prepare(self,owner='alice'):return self.jobs.prepare(owner,self.config,'experiment',3)
    def test_validation_has_no_training_and_exact_retry_is_idempotent(self):
        with patch('subprocess.Popen',side_effect=AssertionError('prepare cannot execute')):p=self.prepare()
        first=self.jobs.submit('alice',p['plan_sha256'],'key',start=False);second=self.jobs.submit('alice',p['plan_sha256'],'key',start=False)
        self.assertEqual(first['job']['job_id'],second['job']['job_id']);self.assertEqual(self.jobs.calls,0)
        changed=self.jobs.prepare('alice',dict(self.config,epochs=3),'experiment',4)
        with self.assertRaisesRegex(JobError,'IDEMPOTENCY_CONFLICT'):self.jobs.submit('alice',changed['plan_sha256'],'key',start=False)
        self.jobs.run_job('alice',first['job']['job_id']);self.assertEqual(self.jobs.calls,1)
        self.assertEqual(self.jobs.submit('alice',p['plan_sha256'],'key')['job']['status'],'succeeded');self.assertEqual(self.jobs.calls,1)
    def test_cross_identity_unknown_plan_and_busy(self):
        p=self.prepare()
        with self.assertRaisesRegex(JobError,'UNKNOWN_PLAN'):self.jobs.submit('bob',p['plan_sha256'],'key',start=False)
        j=self.jobs.submit('alice',p['plan_sha256'],'key',start=False)['job']
        with self.assertRaisesRegex(JobError,'UNKNOWN_JOB'):self.jobs.observe('bob',j['job_id'])
        with self.assertRaisesRegex(JobError,'ACCESS_DENIED'):self.prepare('outsider')
        p2=self.prepare('bob')
        with self.assertRaisesRegex(JobError,'BUSY'):self.jobs.submit('bob',p2['plan_sha256'],'new',start=False)
    def test_concurrent_duplicate_admits_once(self):
        p=self.prepare();ids=[]
        def submit():ids.append(self.jobs.submit('alice',p['plan_sha256'],'same',start=False)['job']['job_id'])
        ts=[threading.Thread(target=submit) for _ in range(4)]
        for t in ts:t.start()
        for t in ts:t.join()
        self.assertEqual(len(ids),4);self.assertEqual(len(set(ids)),1)
    def test_restart_preserves_job_but_does_not_retry_uncertain_execution(self):
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'retry',start=False)['job'];old=self.jobs.backend.boot
        self.jobs.close();self.jobs=SimulatedJobs(Path(self.tmp.name)/'store',principals={'alice','bob'})
        observed=self.jobs.observe('alice',j['job_id']);self.assertNotEqual(observed['worker_boot_id'],old)
        self.assertEqual(observed['job']['status'],'recovery_required');self.assertEqual(observed['job']['submitted_worker_boot_id'],old)
        self.assertEqual(self.jobs.submit('alice',p['plan_sha256'],'retry')['job']['job_id'],j['job_id']);self.assertEqual(self.jobs.calls,0)
        with self.assertRaisesRegex(JobError,'STALE_WORKER'):self.jobs.submit('alice',p['plan_sha256'],'new',start=False)
        p2=self.prepare()
        with self.assertRaisesRegex(JobError,'BUSY'):self.jobs.submit('alice',p2['plan_sha256'],'new',start=False)
    def test_completion_artifacts_hashes_access_and_failure(self):
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'key',start=False)['job'];self.jobs.run_job('alice',j['job_id'])
        observed=self.jobs.observe('alice',j['job_id']);self.assertEqual(observed['job']['execution_kind'],'contract_simulation')
        a=observed['job']['artifacts'][0];self.assertNotIn('file',a)
        result=self.jobs.artifact('alice',j['job_id'],a['artifact_id']);self.assertEqual(result['sha256'],a['sha256'])
        with self.assertRaisesRegex(JobError,'UNKNOWN_JOB'):self.jobs.artifact('bob',j['job_id'],a['artifact_id'])
        with self.assertRaisesRegex(JobError,'UNKNOWN_ARTIFACT'):self.jobs.artifact('alice',j['job_id'],'../../private')
        (self.jobs.store/j['job_id']/'validated_plan.json').write_text('changed')
        with self.assertRaisesRegex(JobError,'ARTIFACT_CHANGED'):self.jobs.artifact('alice',j['job_id'],a['artifact_id'])
        p2=self.prepare();j2=self.jobs.submit('alice',p2['plan_sha256'],'fail',start=False)['job']
        with patch.object(self.jobs,'_execute',return_value=0):self.jobs.run_job('alice',j2['job_id'])
        self.assertEqual(self.jobs.observe('alice',j2['job_id'])['job']['status'],'failed')
    def test_recheck_before_execute_and_reject_changed_plan(self):
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'key',start=False)['job']
        with patch.object(self.jobs.backend,'capabilities',return_value={'sha256':'0'*64}):self.jobs.run_job('alice',j['job_id'])
        self.assertEqual(self.jobs.observe('alice',j['job_id'])['job']['error_code'],'CAPABILITIES_CHANGED');self.assertEqual(self.jobs.calls,0)
        p=self.prepare();body=copy.deepcopy(p['plan']);body['seed']=999
        with self.jobs.db:self.jobs.db.execute('UPDATE plans SET body=? WHERE hash=?',(json.dumps(body),p['plan_sha256']))
        with self.assertRaisesRegex(JobError,'PLAN_CHANGED'):self.jobs.submit('alice',p['plan_sha256'],'new',start=False)
    def test_command_uses_bounded_workflow_and_resource_gate(self):
        p=self.prepare();cmd=self.jobs._command(p['plan'],self.jobs.store/'example')
        self.assertIn('svg_single_document_v1.json',' '.join(cmd));self.assertIn('--epochs',cmd);self.assertNotIn('shell',cmd)
        with patch.object(self.jobs,'_resources',side_effect=JobError('RESOURCES_UNAVAILABLE')):
            with self.assertRaisesRegex(JobError,'RESOURCES_UNAVAILABLE'):self.jobs.submit('alice',p['plan_sha256'],'key')
        with self.assertRaisesRegex(JobError,'EPOCHS'):self.jobs.prepare('alice',dict(self.config,epochs=1),'experiment',0)


class IndependentJobConsumerTest(unittest.TestCase):
    def test_terminal_state_correlation_and_out_of_order(self):
        from server.training.job_consumer import observe,InvalidObservation
        with tempfile.TemporaryDirectory() as tmp:
            jobs=SimulatedJobs(Path(tmp)/'store',principals={'alice'})
            try:
                helper=validation_tests.TrainingContractTest();helper.setUp();config=helper.request(jobs.backend)['config']
                p=jobs.prepare('alice',config,'experiment',3);first=jobs.submit('alice',p['plan_sha256'],'key',start=False);jid=first['job']['job_id']
                expected=dict(worker_boot_id=jobs.backend.boot,owner='alice',job_id=jid,plan_sha256=p['plan_sha256'],experiment_id='experiment',draft_revision=3,allow_simulation=True)
                observe(first,**expected);jobs.run_job('alice',jid);done=jobs.observe('alice',jid);observe(done,last=first,**expected)
                with self.assertRaises(InvalidObservation):observe(first,last=done,**expected)
                with self.assertRaises(InvalidObservation):observe(done,**dict(expected,owner='bob'))
                with self.assertRaises(InvalidObservation):observe(done,**dict(expected,allow_simulation=False))
                broken=copy.deepcopy(done);broken['job']['artifacts']=[]
                with self.assertRaises(InvalidObservation):observe(broken,**expected)
            finally:jobs.close()

if __name__=='__main__':unittest.main()
