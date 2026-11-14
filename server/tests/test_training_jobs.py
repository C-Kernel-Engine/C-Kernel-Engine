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
        checks['learning'].update(last_epoch_loss=1.0,heldout_loss_after=1.5)
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
    def test_progress_is_persisted_monotonic_and_client_observable(self):
        from server.training.job_consumer import observe
        import time
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'progress',start=False)['job']
        with self.jobs.guard,self.jobs.db:
            row=self.jobs._read('alice',j['job_id']);row.update(status='running',stage='workflow',updated_at=time.time());self.jobs._save(row)
        first=self.jobs.observe('alice',j['job_id'])
        path=self.jobs.store/j['job_id']/'run/progress.json'
        path.parent.mkdir(parents=True)
        atomic_json(path,dict(stage='training',completed_steps=1,total_steps=4))
        second=self.jobs.observe('alice',j['job_id'])
        self.assertGreater(second['job']['updated_at'],first['job']['updated_at'])
        observe(second,worker_boot_id=self.jobs.backend.boot,owner='alice',job_id=j['job_id'],plan_sha256=p['plan_sha256'],experiment_id='experiment',draft_revision=3,last=first,allow_simulation=True)
        self.assertEqual(self.jobs._read('alice',j['job_id'])['progress'],second['job']['progress'])
        for bad in (dict(stage='training',completed_steps=0,total_steps=4),dict(stage='training',completed_steps=2,total_steps=5),[],dict(stage='complete',completed_steps=2,total_steps=4)):
            atomic_json(path,bad)
            self.assertEqual(self.jobs.observe('alice',j['job_id'])['job'],second['job'])

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
    def test_incomplete_success_report_and_completed_restart(self):
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'bad-report',start=False)['job']
        def incomplete(command,folder):atomic_json(folder/'report.json',dict(status='PASS',passed=True,checks={}));return 0
        with patch.object(self.jobs,'_execute',incomplete):self.jobs.run_job('alice',j['job_id'])
        self.assertEqual(self.jobs.observe('alice',j['job_id'])['job']['status'],'failed')
        good=self.jobs.submit('alice',p['plan_sha256'],'good-report',start=False)['job'];self.jobs.run_job('alice',good['job_id'])
        self.jobs.close();self.jobs=SimulatedJobs(Path(self.tmp.name)/'store',principals={'alice','bob'})
        self.assertEqual(self.jobs.observe('alice',good['job_id'])['job']['status'],'succeeded')
        self.assertEqual(self.jobs.submit('alice',p['plan_sha256'],'good-report')['job']['job_id'],good['job_id'])
        self.assertEqual(self.jobs.calls,0)

    def test_recheck_before_execute_and_reject_changed_plan(self):
        p=self.prepare();j=self.jobs.submit('alice',p['plan_sha256'],'key',start=False)['job']
        with patch.object(self.jobs.backend,'capabilities',return_value={'sha256':'0'*64}):self.jobs.run_job('alice',j['job_id'])
        self.assertEqual(self.jobs.observe('alice',j['job_id'])['job']['error_code'],'CAPABILITIES_CHANGED');self.assertEqual(self.jobs.calls,0)
        p=self.prepare();body=copy.deepcopy(p['plan']);body['seed']=999
        with self.jobs.db:self.jobs.db.execute('UPDATE plans SET body=? WHERE hash=?',(json.dumps(body),p['plan_sha256']))
        with self.assertRaisesRegex(JobError,'PLAN_CHANGED'):self.jobs.submit('alice',p['plan_sha256'],'new',start=False)
    def test_expired_plan_and_store_exclusivity(self):
        p=self.prepare()
        with patch('server.training.jobs.time.time',return_value=p['plan']['expires_at']+1):
            with self.assertRaisesRegex(JobError,'PLAN_EXPIRED'):self.jobs.submit('alice',p['plan_sha256'],'expired',start=False)
        with self.assertRaisesRegex(JobError,'STORE_BUSY'):SimulatedJobs(self.jobs.store,principals={'alice'})

    def test_interpreter_symlink_keeps_environment_identity(self):
        import sys
        alias=Path(self.tmp.name)/'venv/bin/python';alias.parent.mkdir(parents=True);alias.symlink_to(sys.executable)
        other=SimulatedJobs(Path(self.tmp.name)/'other',principals={'alice'},python=alias)
        try:
            self.assertEqual(other.python,str(alias))
            p=other.prepare('alice',self.config,'experiment',1)
            self.assertEqual(other._command(p['plan'],other.store/'job')[0],str(alias))
        finally:other.close()

    def test_command_uses_bounded_workflow_and_resource_gate(self):
        p=self.prepare();cmd=self.jobs._command(p['plan'],self.jobs.store/'example')
        self.assertIn('svg_single_document_v1.json',' '.join(cmd));self.assertIn('--epochs',cmd);self.assertNotIn('shell',cmd)
        with patch.object(self.jobs,'_resources',side_effect=JobError('RESOURCES_UNAVAILABLE')):
            with self.assertRaisesRegex(JobError,'RESOURCES_UNAVAILABLE'):self.jobs.submit('alice',p['plan_sha256'],'key')
        with self.assertRaisesRegex(JobError,'EPOCHS'):self.jobs.prepare('alice',dict(self.config,epochs=1),'experiment',0)


class IndependentJobConsumerTest(unittest.TestCase):
    def test_versioned_simulated_job_fixture(self):
        from server.training.job_consumer import observe,InvalidObservation
        from server.training.contract import ROOT
        f=json.loads((ROOT/'server/training/fixtures/job_v1.json').read_text());j=f['completed']['job']
        expected=dict(worker_boot_id=f['completed']['worker_boot_id'],owner=j['owner'],job_id=j['job_id'],plan_sha256=j['plan_sha256'],experiment_id=j['experiment_id'],draft_revision=j['draft_revision'])
        observe(f['completed'],last=f['queued'],allow_simulation=True,**expected)
        with self.assertRaises(InvalidObservation):observe(f['completed'],**expected)

    def test_terminal_state_correlation_and_out_of_order(self):
        from server.training.job_consumer import observe,verify_artifacts,InvalidObservation
        with tempfile.TemporaryDirectory() as tmp:
            jobs=SimulatedJobs(Path(tmp)/'store',principals={'alice'})
            try:
                helper=validation_tests.TrainingContractTest();helper.setUp();config=helper.request(jobs.backend)['config']
                p=jobs.prepare('alice',config,'experiment',3);first=jobs.submit('alice',p['plan_sha256'],'key',start=False);jid=first['job']['job_id']
                expected=dict(worker_boot_id=jobs.backend.boot,owner='alice',job_id=jid,plan_sha256=p['plan_sha256'],experiment_id='experiment',draft_revision=3,allow_simulation=True)
                observe(first,**expected);jobs.run_job('alice',jid);done=jobs.observe('alice',jid);observe(done,last=first,**expected)
                self.assertEqual(len(verify_artifacts(done,lambda a,o:jobs.artifact('alice',jid,a,o))),2)
                def corrupt(a,o):
                    r=jobs.artifact('alice',jid,a,o);r['base64']='eA==';return r
                with self.assertRaises(InvalidObservation):verify_artifacts(done,corrupt)
                with self.assertRaises(InvalidObservation):observe(first,last=done,**expected)
                with self.assertRaises(InvalidObservation):observe(done,**dict(expected,owner='bob'))
                with self.assertRaises(InvalidObservation):observe(done,**dict(expected,allow_simulation=False))
                broken=copy.deepcopy(done);broken['job']['artifacts']=[]
                with self.assertRaises(InvalidObservation):observe(broken,**expected)
            finally:jobs.close()

class RealSvgJobTest(unittest.TestCase):
    def test_generated_workflow_job(self):
        if os.environ.get('CKE_RUN_SVG_JOB_ACCEPTANCE')!='1':self.skipTest('real generated SVG job disabled: NOT_TESTED')
        from version.v8.scripts.run_svg_job_acceptance_v8 import run
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            result=run(Path(tmp)/'store',sys.executable)
            self.assertEqual(result['job']['status'],'succeeded');self.assertTrue(result['duplicate_same_job'])

if __name__=='__main__':unittest.main()
