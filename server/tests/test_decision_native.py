"""Real generated head packages over loopback; synthetic feature oracle is separate."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from server.decision import contract as wire
from server.decision.consumer import Consumer,LoopbackHTTP,validate
from server.decision.native import NativeHead
from server.decision.native_server import NativeServer

ROOT=Path(__file__).resolve().parents[2]

class NativeDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        try:import torch
        except ImportError:raise unittest.SkipTest('PyTorch unavailable: independent nonzero head oracle NOT_TESTED')
        from tests.test_v8_clef_flash_complete_head import synthetic_weights,reference
        from version.v8.decision.package_clef_head import build_package
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup)
        cls.folder=Path(cls.tmp.name)/'package';cls.raw=synthetic_weights()
        cls.inputs=[];artifacts=[];cls.expected=[]
        for i in range(2):
            x=dict(hidden=np.random.default_rng(i+2).normal(0,.2,(6,8)).astype('f4'),input_ids=np.arange(1,7,dtype='i4'),question_spans=np.array([[0,2],[3,6]],dtype='i4'),option_spans=np.array([[1,3],[2,5],[5,6]],dtype='i4'),global_span=np.array([[5,6]],dtype='i4'),option_counts=np.array([2,1],dtype='i4'),question_type_ids=np.array([0,2],dtype='i4'))
            cls.inputs.append(x);artifacts.append(dict(id=f'features-{i}',inputs=x,question_index=0,candidate_ids=['read-all','clarify']))
            cls.expected.append(reference(x['hidden'],x['input_ids'],x['question_spans'],x['option_spans'],x['option_counts'],x['question_type_ids'],cls.raw)['option_logits'].ravel()[:2])
        cls.identity=build_package(cls.folder,cls.raw,artifacts,dict(id='synthetic-head',revision='0'*40),heads=2,provenance=dict(features='synthetic',embedding='synthetic'))
        cls.manifest=cls.folder/'package.json'
    def setUp(self):
        self.head=NativeHead(self.manifest,self.identity);self.addCleanup(self.head.close)
        self.server=NativeServer(self.head);self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.addCleanup(self.stop)
        self.exchange=LoopbackHTTP(f'http://127.0.0.1:{self.server.server_port}/evaluate')
    def stop(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def run_request(self,i=0):return Consumer().evaluate(self.head.request(f'features-{i}'),self.exchange)
    def test_native_oracle_and_reused_a_b_a(self):
        import numpy as np
        first=None
        for i in (0,1,0):
            result=self.run_request(i);scores=[x['logit'] for x in result['scores']]
            np.testing.assert_allclose(scores,self.expected[i],atol=2e-6,rtol=0)
            if first is None:first=scores
            elif i==0:self.assertEqual(scores,first)
            self.assertEqual(result['implementation_stage'],'head_only');self.assertEqual(result['evidence']['parity']['status'],'NOT_TESTED')
            self.assertEqual(result['evidence']['artifact_id'],self.identity)
        self.assertNotEqual(list(self.expected[0]),list(self.expected[1]))
        fresh=NativeHead(self.manifest,self.identity)
        try:
            q=fresh.request('features-0');status,r=fresh.evaluate(q);self.assertEqual(status,200)
            self.assertEqual([x['logit'] for x in validate(r,q)['scores']],first)
        finally:fresh.close()
    def test_permutation_rejection_and_recovery(self):
        q=self.head.request('features-0');q['candidate_ids'].reverse()
        result=Consumer().evaluate(q,self.exchange)
        self.assertEqual([x['candidate_id'] for x in result['scores']],q['candidate_ids'])
        for mutate,code in [(lambda q:q['input'].update(sha256='0'*64),'UNKNOWN_ARTIFACT'),(lambda q:q.update(worker_boot_id='0'*32),'RESTARTED'),(lambda q:q.update(candidate_ids=['wrong']),'CANDIDATE_MISMATCH')]:
            q=self.head.request('features-0');mutate(q);status,r=self.head.evaluate(q)
            self.assertFalse(r['ok']);self.assertEqual(r['error']['code'],code)
        self.assertEqual(self.run_request()['execution_kind'],'native_generated')
    def test_busy_deadline_and_recovery(self):
        original=self.head.execute;entered=threading.Event();release=threading.Event();results=[]
        def delayed(*args):entered.set();release.wait(2);return original(*args)
        q=self.head.request('features-0');q['timeout_ms']=200
        with patch.object(self.head,'execute',delayed):
            thread=threading.Thread(target=lambda:results.append(self.head.evaluate(q)));thread.start()
            try:
                self.assertTrue(entered.wait(1));status,r=self.head.evaluate(self.head.request('features-1'))
                self.assertEqual(status,429);self.assertEqual(r['error']['code'],'BUSY');time.sleep(.21)
            finally:release.set();thread.join()
        self.assertEqual(results[0][1]['error']['code'],'DEADLINE_EXCEEDED')
        self.assertEqual(self.run_request()['execution_kind'],'native_generated')
    def test_cancelled_native_result_never_publishes(self):
        from server.decision.consumer import InvalidResult
        original=self.head.execute;entered=threading.Event();release=threading.Event();outcomes=[];consumer=Consumer()
        def held(*args):entered.set();release.wait(2);return original(*args)
        def run():
            try:consumer.evaluate(self.head.request('features-0'),self.exchange)
            except InvalidResult:outcomes.append('rejected')
        with patch.object(self.head,'execute',held):
            thread=threading.Thread(target=run);thread.start()
            try:
                self.assertTrue(entered.wait(1));consumer.cancel()
                self.assertIsNone(consumer.current)
                self.assertEqual(self.head.evaluate(self.head.request('features-1'))[0],429)
            finally:release.set();thread.join()
        self.assertEqual(outcomes,['rejected']);self.assertIsNone(consumer.current)
        self.assertEqual(consumer.evaluate(self.head.request('features-1'),self.exchange)['execution_kind'],'native_generated')

    def test_close_waits_for_native_ownership(self):
        original=self.head.execute;entered=threading.Event();release=threading.Event();closed=threading.Event()
        def held(*args):entered.set();release.wait(2);return original(*args)
        self.head.execute=held
        thread=threading.Thread(target=lambda:self.head.evaluate(self.head.request('features-0')));thread.start()
        closer=threading.Thread(target=lambda:(self.head.close(),closed.set()))
        try:
            self.assertTrue(entered.wait(1));closer.start()
            self.assertFalse(closed.wait(.02));self.assertIsNotNone(self.head.library)
        finally:release.set();thread.join();closer.join()
        self.assertTrue(closed.is_set());self.assertIsNone(self.head.library)
        status,r=self.head.evaluate(self.head.request('features-0'))
        self.assertEqual(status,500);self.assertIsNone(r['result'])

    def test_weights_unchanged_and_library_sealed(self):
        import ctypes
        import os
        def digest():
            h=hashlib.sha256()
            for w in self.head.manifest['weight_spans']:h.update(ctypes.string_at(self.head.address+w['arena_offset'],w['size']))
            return h.hexdigest()
        baseline=digest()
        for i in (0,1,0):self.run_request(i)
        self.assertEqual(digest(),baseline)
        with self.assertRaises(OSError):os.pwrite(self.head.fd,b'x',0)

    def test_kernel_failure_no_publication_and_recovery(self):
        import ctypes
        w=self.head.manifest['weight_spans'][-1];ptr=self.head.address+w['arena_offset'];saved=ctypes.string_at(ptr,4)
        try:
            ctypes.c_float.from_address(ptr).value=float('nan')
            status,r=self.head.evaluate(self.head.request('features-0'));self.assertFalse(r['ok']);self.assertIsNone(r['result'])
            from server.decision.consumer import InvalidResult
            with self.assertRaisesRegex(InvalidResult,'REMOTE_INTERNAL_ERROR'):self.run_request()
        finally:ctypes.memmove(ptr,saved,4)
        self.run_request()
    def test_reject_tampered_packages_before_load(self):
        original=json.loads(self.manifest.read_text())
        def offset(m):m['inputs']['hidden']['offset']=m['weight_spans'][0]['arena_offset']
        def inputs(m):m['artifacts'][0]['input_sha256']='0'*64
        def wrong_global(m):m['inputs']['global_span']['shape']=[1,3]
        for mutate in (offset,inputs,wrong_global,lambda m:m['files']['library'].update(sha256='0'*64)):
            with self.subTest(mutate=mutate):
                m=copy.deepcopy(original);mutate(m);data=wire.encode(m)
                path=self.folder/'invalid.json';path.write_bytes(data)
                with patch('server.decision.native.ctypes.CDLL') as load:
                    with self.assertRaisesRegex(ValueError,'INVALID_PACKAGE'):NativeHead(path,hashlib.sha256(data).hexdigest())
                    load.assert_not_called()
        with self.assertRaises(ValueError):NativeHead(self.manifest,'0'*64)
    def test_malformed_http_never_executes(self):
        import http.client
        bodies=[b'{"private":"not-a-request"}', b'{"schema":1,"schema":2}', b'x'*16385]
        with patch.object(self.head,'execute') as execute:
            for body in bodies:
                conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
                try:
                    conn.request('POST','/evaluate',body,{'Content-Type':'application/json'})
                    response=conn.getresponse();self.assertEqual(response.status,400)
                    self.assertEqual(json.loads(response.read()),{'error':{'code':'INVALID_REQUEST'}})
                finally:conn.close()
            execute.assert_not_called()
        self.run_request()

    def test_deployed_worker_without_site_packages(self):
        process=subprocess.Popen([sys.executable,'-S','-m','server.decision.native_server','--package',str(self.manifest),'--package-sha256',self.identity],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            import select
            self.assertTrue(select.select([process.stdout],[],[],10)[0],'native worker startup timed out')
            line=process.stdout.readline();self.assertTrue(line)
            info=json.loads(line);result=Consumer().evaluate(info['requests'][0],LoopbackHTTP(info['url']))
            self.assertEqual(result['execution_kind'],'native_generated')
        finally:
            process.terminate();process.communicate(timeout=5)


class PinnedNativeDecisionTest(unittest.TestCase):
    def test_pinned_head_over_http(self):
        import os
        source=os.environ.get('CKE_CLEF_FLASH_REFERENCE_SOURCE');asset=os.environ.get('CKE_CLEF_FLASH_HEAD_PATH')
        if not source or not asset:self.skipTest('pinned head/reference unavailable: asset-backed parity NOT_TESTED')
        try:
            import numpy as np
            import torch
            from safetensors.torch import load_file
        except ImportError:self.skipTest('PyTorch/safetensors unavailable: pinned oracle NOT_TESTED')
        import importlib.util
        from types import SimpleNamespace
        from version.v8.decision.package_clef_head import build_package
        identity=json.loads((ROOT/'tests/fixtures/decision/clef_flash/identity.json').read_text())
        for file,key in ((source,'joint_schema_model_sha256'),(asset,'joint_head_weights_sha256')):
            self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(),identity[key])
        spec=importlib.util.spec_from_file_location('clef_native_reference',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        state=load_file(asset);ref=module.JointSchemaHead(hidden_size=4096,width=1024,routing_layers=2,layers=4,heads=16,feedforward=4096,dropout=0)
        ref.load_state_dict(state,strict=True);ref.eval()
        raw={k:v.float().numpy() for k,v in state.items()}
        raw['output_embedding.weight']=np.random.default_rng(113).normal(0,.02,(16,4096)).astype('f4')
        artifacts=[];expected=[]
        for i in range(2):
            x=dict(hidden=np.random.default_rng(48+i).normal(0,.2,(6,4096)).astype('f4'),input_ids=np.arange(1,7,dtype='i4'),question_spans=np.array([[0,2],[3,6]],dtype='i4'),option_spans=np.array([[1,3],[2,5],[5,6]],dtype='i4'),global_span=np.array([[5,6]],dtype='i4'),option_counts=np.array([2,1],dtype='i4'),question_type_ids=np.array([0,2],dtype='i4'))
            artifacts.append(dict(id=f'pinned-{i}',inputs=x,question_index=0,candidate_ids=['read-all','clarify']))
            questions=[SimpleNamespace(question_span=tuple(x['question_spans'][0]),question_type=0,option_spans=[tuple(s) for s in x['option_spans'][:2]]),SimpleNamespace(question_span=tuple(x['question_spans'][1]),question_type=2,option_spans=[tuple(x['option_spans'][2])])]
            with torch.no_grad():expected.append(ref(torch.from_numpy(x['hidden'])[None],torch.from_numpy(x['input_ids'].astype('int64'))[None],torch.ones((1,6),dtype=torch.int64),[SimpleNamespace(questions=questions)],torch.from_numpy(raw['output_embedding.weight']))[0][0].numpy())
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'package'
            digest=build_package(folder,raw,artifacts,dict(id='clef-flash-head',revision=identity['revision']),heads=16,provenance=dict(head_weights_sha256=identity['joint_head_weights_sha256'],reference_source_sha256=identity['joint_schema_model_sha256'],features='synthetic',embedding='synthetic'))
            head=NativeHead(folder/'package.json',digest)
            server=NativeServer(head);thread=threading.Thread(target=server.serve_forever);thread.start()
            try:
                transport=LoopbackHTTP(f'http://127.0.0.1:{server.server_port}/evaluate');previous=None
                for i in (0,1,0):
                    q=head.request(f'pinned-{i}');result=Consumer().evaluate(q,transport)
                    actual=np.array([s['logit'] for s in result['scores']]);err=actual-expected[i]
                    np.testing.assert_allclose(actual,expected[i],atol=2e-6,rtol=0)
                    if i==0 and previous is not None:np.testing.assert_array_equal(actual,previous)
                    if i==0:previous=actual
                    print('pinned native package',json.dumps(dict(package_sha256=digest,library_sha256=result['evidence']['library_sha256'],artifact=i,max_abs=float(abs(err).max()),rmse=float(np.sqrt(np.mean(err**2))),latency_ms=result['latency_ms'],scope='head_only',features='synthetic',embedding='synthetic')))
            finally:server.shutdown();server.server_close();thread.join();head.close()

if __name__=='__main__':unittest.main()
