import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from server.vision_http import Worker, create_app, validate

IMAGE=Path(__file__).resolve().parents[2]/'tests/fixtures/vision_session/two_colors.png'
def body():
    return {'model':'test','max_tokens':4,'messages':[{'role':'user','content':[
        {'type':'text','text':'Before'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(IMAGE.read_bytes()).decode()}},
        {'type':'text','text':'After'}]}]}

FAKE='''import json,sys
p=json.load(sys.stdin)
if p.get('probe'): print(json.dumps({'type':'complete','capabilities':{'vision':True}}))
else:
 print(json.dumps({'type':'delta','text':'Blue'}))
 print(json.dumps({'type':'complete','finish_reason':'length','usage':{'prompt_tokens':20,'completion_tokens':1,'total_tokens':21}}))
'''

class VisionHTTPTests(unittest.TestCase):
    def test_order_and_followup(self):
        b=body();b['messages'] += [{'role':'assistant','content':'Blue'},{'role':'user','content':'Again?'}]
        messages,image,budget=validate(b,'test','<|image_pad|>')
        self.assertEqual([p['type'] for p in messages[0]['content']],['text','image','text'])
        self.assertEqual(messages[2]['content'][0]['text'],'Again?')
        self.assertEqual(image.data,IMAGE.read_bytes())
    def test_rejections(self):
        for change in ({'tools':[]},{'temperature':1},{'max_tokens':True},{'model':'other'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate(dict(body(),**change),'test','<|image_pad|>')
        b=body();b['messages'][0]['content'][0]['text']='<|image_pad|>'
        with self.assertRaises(ValueError):validate(b,'test','<|image_pad|>')
    def test_http_stream_and_recovery(self):
        with patch('server.vision_bundle.load',return_value={'model':'test','image_placeholder':'<|image_pad|>'}):
            app=create_app('unused',command=[sys.executable,'-c',FAKE])
        with TestClient(app) as client:
            self.assertTrue(client.get('/health').json()['ready'])
            self.assertEqual(client.post('/v1/chat/completions',json=dict(body(),tools=[])).status_code,400)
            result=client.post('/v1/chat/completions',json=body())
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()['choices'][0]['message']['content'],'Blue')
            result=client.post('/v1/chat/completions',json=dict(body(),stream=True))
            self.assertEqual(result.text.count('[DONE]'),1)
            self.assertIn('"finish_reason": "length"',result.text)
            self.assertFalse(client.get('/health').json()['busy'])
    def test_transport_failure_before_iteration_releases_admission(self):
        with patch('server.vision_bundle.load',return_value={'model':'test','image_placeholder':'<|image_pad|>'}):
            app=create_app('unused',command=[sys.executable,'-c',FAKE])
        failed=False
        async def transport(scope,receive,send):
            async def broken_send(message):
                nonlocal failed
                if scope.get('path')=='/v1/chat/completions' and message['type']=='http.response.start' and not failed:
                    failed=True
                    raise OSError('injected transport disconnect before stream starts')
                await send(message)
            await app(scope,receive,broken_send)
        with TestClient(transport,raise_server_exceptions=False) as client:
            client.post('/v1/chat/completions',json=dict(body(),stream=True))
            self.assertTrue(failed)
            self.assertFalse(client.get('/health').json()['busy'])
            self.assertEqual(client.post('/v1/chat/completions',json=body()).status_code,200)

    def test_reference_replay_is_operator_selected(self):
        fake=FAKE.replace("else:\n print", "else:\n assert p.get('verify_reference') is True\n print",1)
        with patch('server.vision_bundle.load',return_value={'model':'test','image_placeholder':'<|image_pad|>'}):
            app=create_app('unused',command=[sys.executable,'-c',fake],verify_reference=True)
        with TestClient(app) as client:
            self.assertTrue(client.get('/v1/capabilities').json()['reference_replay'])
            self.assertEqual(client.post('/v1/chat/completions',json=body()).status_code,200)
            self.assertEqual(client.post('/v1/chat/completions',json=dict(body(),verify_reference=False)).status_code,400)

    def test_worker_failure_is_not_complete(self):
        async def check():
            worker=Worker('',1,1<<30,[sys.executable,'-c','print("not json")'])
            with self.assertRaises(RuntimeError):
                async for _ in worker.events({}):pass
            self.assertIsNone(worker.process)
        asyncio.run(check())
    def test_cancel_while_process_creation_returns_still_reaps(self):
        async def check():
            original=asyncio.create_subprocess_exec
            created=asyncio.Event();release=asyncio.Event();children=[]
            async def delayed(*args,**kwargs):
                process=await original(*args,**kwargs)
                children.append(process);created.set()
                await release.wait()
                return process
            worker=Worker('',1,1<<30,[sys.executable,'-c','import time;time.sleep(60)'])
            async def consume():
                async for event in worker.events({}):pass
            task=None
            try:
                with patch('server.vision_http.asyncio.create_subprocess_exec',side_effect=delayed):
                    task=asyncio.create_task(consume())
                    await asyncio.wait_for(created.wait(),5)
                    task.cancel();await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,5)
                self.assertIsNotNone(children[0].returncode)
                self.assertIsNone(worker.process)
            finally:
                release.set()
                if task is not None:
                    if not task.done():task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
                for child in children:
                    if child.returncode is None:child.kill()
                    await child.wait()
        asyncio.run(check())

    def test_cancel_reaps_worker_and_recovers(self):
        async def check():
            worker=Worker('',1,1<<30,[sys.executable,'-c','import time; print(\'{"type":"stage"}\',flush=True); time.sleep(60)'])
            events=worker.events({})
            self.assertEqual((await anext(events))['type'],'stage')
            process=worker.process
            await worker.stop()
            self.assertIsNotNone(process.returncode)
            with self.assertRaises(RuntimeError):await anext(events)
            worker.command=[sys.executable,'-c',FAKE]
            result=[e async for e in worker.events({'probe':True})]
            self.assertEqual(result[-1]['type'],'complete')
        asyncio.run(check())

if __name__=='__main__':unittest.main()
