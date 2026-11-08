"""Proposed decision contract over real sockets; fixture replay, NOT inference."""
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

from server.decision import contract as wire
from server.decision.consumer import Consumer, InvalidResult, LoopbackHTTP, validate

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'server/decision/fixtures/evaluate.json'

class FixtureServer:
    """Test-only responder. Never imported by a deployed CKE worker."""
    def __init__(self):
        self.fixture=json.loads(FIXTURE.read_text());self.adapter=uuid.uuid4().hex;self.boot=uuid.uuid4().hex
        self.slot=threading.BoundedSemaphore(1);self.delay=0;self.drip=False;self.entered=threading.Event();self.mutate=lambda r:r
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def setup(self):super().setup();self.connection.settimeout(1)
            def do_POST(self):
                q=None
                try:
                    wire.require(self.path=='/evaluate','UNSUPPORTED_TASK')
                    wire.require(self.headers.get('Content-Type')=='application/json')
                    wire.require(not self.headers.get('Transfer-Encoding') and len(self.headers.get_all('Content-Length',[]))==1)
                    n=int(self.headers.get('Content-Length','0'))
                    wire.require(1<=n<=wire.MAX_BYTES,'MESSAGE_TOO_LARGE')
                    q=wire.decode(self.rfile.read(n));wire.validate_request(q)
                    status,result=owner.evaluate(q)
                    body=wire.encode(owner.mutate(result))
                except (wire.ContractError,ValueError) as exc:
                    code=str(exc) if isinstance(exc,wire.ContractError) else 'INVALID_REQUEST'
                    result={k:None for k in ('request_id','model','request_sha256')}
                    result.update(schema=wire.SCHEMA,adapter_boot_id=owner.adapter,worker_boot_id=owner.boot,ok=False,result=None,error={'code':code})
                    if type(q) is dict:
                        result['request_id']=q.get('request_id') if wire.matches(wire.IDENT,q.get('request_id')) else None
                    status=400;body=wire.encode(result)
                self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers()
                try:
                    if owner.drip:
                        for byte in body:self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.01)
                    else:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        self.http=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.http.daemon_threads=True
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.http.server_port}/evaluate'
    def close(self):self.http.shutdown();self.http.server_close();self.thread.join()
    def request(self):
        f=self.fixture
        return dict(schema=wire.SCHEMA,request_id=uuid.uuid4().hex,adapter_boot_id=self.adapter,worker_boot_id=self.boot,model=f['model'].copy(),task='registered_feature_scores',input=dict(artifact_id=f['artifact_id'],sha256=f['input_sha256']),candidate_ids=list(f['scores']),timeout_ms=1000,policy=dict(minimum_logit_margin=0.))
    def evaluate(self,q):
        started=time.monotonic();f=self.fixture
        base={k:copy.deepcopy(q[k]) for k in ('schema','request_id','adapter_boot_id','worker_boot_id','model')}
        base.update(request_sha256=wire.digest(q),ok=False,result=None,error=None)
        def error(code,status=400):return status,dict(base,error={'code':code})
        if q['adapter_boot_id']!=self.adapter or q['worker_boot_id']!=self.boot:return error('RESTARTED',409)
        if q['model']!=f['model']:return error('UNKNOWN_MODEL')
        if q['input']!={'artifact_id':f['artifact_id'],'sha256':f['input_sha256']}:return error('UNKNOWN_ARTIFACT')
        if set(q['candidate_ids'])!=set(f['scores']):return error('CANDIDATE_MISMATCH')
        if not self.slot.acquire(blocking=False):return error('BUSY',429)
        try:
            self.entered.set();time.sleep(self.delay)
            if (time.monotonic()-started)*1000>q['timeout_ms']:return error('DEADLINE_EXCEEDED',504)
            values=[f['scores'][k] for k in q['candidate_ids']]
            try:decision=wire.rank(q['candidate_ids'],values,q['policy']['minimum_logit_margin'])
            except wire.ContractError:return error('NONFINITE_SCORE')
            result=dict(implementation_stage='contract_simulation',execution_kind='fixture_replay',scores=[dict(candidate_id=k,logit=v) for k,v in zip(q['candidate_ids'],values)],score_semantics='raw_logits',decision=decision,executed=False,latency_ms=(time.monotonic()-started)*1000,evidence=copy.deepcopy(f['evidence']))
            return 200,dict(base,ok=True,result=result)
        finally:self.slot.release()

class DecisionEvaluationTests(unittest.TestCase):
    def setUp(self):self.server=FixtureServer();self.addCleanup(self.server.close);self.exchange=LoopbackHTTP(self.server.url)
    def run_request(self,q=None,consumer=None):return (consumer or Consumer()).evaluate(q or self.server.request(),self.exchange,allow_simulation=True)
    def test_fixture_integrity_and_independent_process_smoke(self):
        self.assertEqual(wire.digest(self.server.fixture['payload']),self.server.fixture['input_sha256'])
        spec=json.loads((ROOT/'server/decision/contract.json').read_text())
        self.assertEqual(spec['limits']['request_bytes'],wire.MAX_BYTES)
        self.assertEqual(spec['limits']['candidate_ids'],wire.MAX_CANDIDATES)
        self.assertFalse(spec['implemented_boundary']['native_head_http_worker'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'request.json';p.write_bytes(wire.encode(self.server.request()))
            result=subprocess.run([sys.executable,'-m','server.decision.consumer','--url',self.server.url,'--request',str(p),'--allow-simulation'],cwd=ROOT,capture_output=True,text=True,timeout=3)
            self.assertEqual(result.returncode,0,result.stderr);report=json.loads(result.stdout)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['result']['execution_kind'],'fixture_replay')
            self.assertEqual(report['result']['evidence']['parity']['status'],'NOT_TESTED')
    def test_order_stable_ids_and_a_b_a(self):
        consumer=Consumer();q=self.server.request();a=self.run_request(q,consumer)
        b=copy.deepcopy(q);b['request_id']='request-b';b['candidate_ids'].reverse();br=self.run_request(b,consumer)
        self.assertEqual([x['candidate_id'] for x in br['scores']],b['candidate_ids'])
        self.assertEqual({x['candidate_id']:x['logit'] for x in a['scores']},{x['candidate_id']:x['logit'] for x in br['scores']})
        again=self.run_request(q,consumer);self.assertEqual(a['scores'],again['scores'])
        self.assertFalse(again['executed'])
    def test_simulation_rejected_without_opt_in(self):
        c=Consumer()
        with self.assertRaisesRegex(InvalidResult,'SIMULATION_NOT_ALLOWED'):c.evaluate(self.server.request(),self.exchange)
        self.assertIsNone(c.current)
    def test_invalid_requests_and_unknown_artifacts(self):
        for field,value in [('candidate_ids',[]),('candidate_ids',['x','x']),('candidate_ids',['x']*17),('request_id',''),('timeout_ms',True),('timeout_ms',1001),('task','execute')]:
            q=self.server.request();q[field]=value
            with self.assertRaises(wire.ContractError):wire.validate_request(q)
        q=self.server.request();q['input']['shape']=[2,-1]
        with self.assertRaises(wire.ContractError):wire.validate_request(q)
        for raw in (b' '*16385,b'{"x":1,"x":2}',b'{"x":NaN}',b'\xff'):
            with self.assertRaises(wire.ContractError):wire.decode(raw)
        for field,value,error in [('model',{'id':'other','revision':'0'*40},'UNKNOWN_MODEL'),('input',{'artifact_id':'other','sha256':'0'*64},'UNKNOWN_ARTIFACT'),('candidate_ids',['other'],'CANDIDATE_MISMATCH')]:
            q=self.server.request();q[field]=value
            with self.assertRaisesRegex(InvalidResult,error):self.run_request(q)
    def test_missing_metadata_cannot_pass_by_matching_nulls(self):
        q=self.server.request();r=self.server.evaluate(q)[1]
        q['input']['sha256']=None;r['result']['evidence']['input_sha256']=None;r['request_sha256']=wire.digest(q)
        with self.assertRaises(InvalidResult):validate(r,q,True)
        c=Consumer();self.run_request(consumer=c)
        with self.assertRaises(InvalidResult):c.evaluate(q,lambda _:self.fail('invalid request reached transport'),True)
        self.assertIsNone(c.current)

    def test_real_socket_rejection_is_structured(self):
        import http.client
        for raw in (b' '*16385,b'{"x":NaN}',b'{}',wire.encode(dict(self.server.request(),task='execute'))):
            conn=http.client.HTTPConnection('127.0.0.1',self.server.http.server_port,timeout=1)
            try:
                conn.request('POST','/evaluate',raw,{'Content-Type':'application/json'})
                response=conn.getresponse();self.assertEqual(response.status,400)
                error=json.loads(response.read());self.assertFalse(error['ok']);self.assertIsNone(error['result']);self.assertIn('code',error['error'])
            finally:conn.close()

    def test_ties_policy_and_nonfinite(self):
        self.server.fixture['scores']={k:1. for k in self.server.fixture['scores']}
        result=self.run_request();self.assertEqual(result['decision']['reason'],'tie')
        self.server.fixture['scores']['read-all']=1.1;q=self.server.request();q['policy']['minimum_logit_margin']=.2
        self.assertEqual(self.run_request(q)['decision']['reason'],'policy_margin')
        self.server.fixture['scores']['read-all']=float('nan')
        with self.assertRaisesRegex(InvalidResult,'NONFINITE_SCORE'):self.run_request()
    def test_malformed_responses_clear_current(self):
        c=Consumer();self.run_request(consumer=c)
        for mutate in (lambda r:r.pop('request_sha256'),lambda r:r['result']['scores'][0].update(candidate_id='wrong'),lambda r:r['result'].update(executed=True),lambda r:r['result']['evidence'].update(input_sha256='0'*64),lambda r:r['result']['evidence']['parity'].update(status='PASS'),lambda r:r['result']['decision'].update(candidate_id='clarify'),lambda r:r.update(worker_boot_id='0'*32),lambda r:r['result']['scores'][0].update(logit=10**400)):
            def altered(r):mutate(r);return r
            self.server.mutate=altered
            with self.assertRaises(InvalidResult):self.run_request(consumer=c)
            self.assertIsNone(c.current)
    def test_restarts_and_refresh(self):
        c=Consumer()
        for which in ('boot','adapter'):
            q=self.server.request();self.run_request(q,c);setattr(self.server,which,uuid.uuid4().hex)
            with self.assertRaisesRegex(InvalidResult,'RESTARTED'):self.run_request(q,c)
            self.assertIsNone(c.current);self.run_request(consumer=c)
    def test_replacement_keeps_new_result_when_old_reply_arrives(self):
        c=Consumer();ready=threading.Event();release=threading.Event();errors=[]
        old=self.server.request();new=self.server.request();new['candidate_ids'].reverse()
        def delayed(q):ready.set();release.wait(2);return self.exchange(q)
        def run():
            try:c.evaluate(old,delayed,True)
            except InvalidResult as e:errors.append(str(e))
        t=threading.Thread(target=run);t.start();self.assertTrue(ready.wait(1))
        result=c.evaluate(new,self.exchange,True);release.set();t.join(2)
        self.assertEqual(errors,['STALE_RESULT']);self.assertEqual(c.current,result)

    def test_slow_body_obeys_total_deadline(self):
        self.server.drip=True;q=self.server.request();q['timeout_ms']=50;c=Consumer();start=time.monotonic()
        with self.assertRaises(InvalidResult):c.evaluate(q,self.exchange,True)
        self.assertLess(time.monotonic()-start,.5);self.assertIsNone(c.current)

    def test_busy_timeout_cancellation_and_late_reply(self):
        c=Consumer();q=self.server.request();self.server.delay=.15;errors=[]
        def run():
            try:self.run_request(q,c)
            except InvalidResult as e:errors.append(str(e))
        t=threading.Thread(target=run);t.start();self.assertTrue(self.server.entered.wait(1))
        with self.assertRaisesRegex(InvalidResult,'BUSY'):self.run_request()
        c.cancel();t.join(2);self.assertFalse(t.is_alive());self.assertEqual(errors,['STALE_RESULT']);self.assertIsNone(c.current)
        q=self.server.request();q['timeout_ms']=10
        with self.assertRaises(InvalidResult):self.run_request(q,c)
        self.assertIsNone(c.current)
        time.sleep(.17);self.server.delay=0;self.run_request(consumer=c)
        # Independent consumer deadline enforcement even if a peer lies.
        q=self.server.request();q['timeout_ms']=1
        def late(q):time.sleep(.01);return self.server.evaluate(q)[1]
        with self.assertRaisesRegex(InvalidResult,'DEADLINE_EXCEEDED'):c.evaluate(q,late,True)
        self.assertIsNone(c.current)

if __name__=='__main__':unittest.main(verbosity=2)
