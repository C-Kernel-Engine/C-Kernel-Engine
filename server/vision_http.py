"""Experimental, loopback-only, one-request-at-a-time vision HTTP adapter."""
import argparse
import anyio
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import signal
import sys
import tempfile
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server.vision_input import prepare_vision_input, ImagePart, TextPart, VisionInputError


def validate(body, model, marker):
    if not isinstance(body, dict) or set(body) - {'model','messages','max_tokens','stream','temperature'}:
        raise ValueError('unsupported request fields; tools are not supported')
    if body.get('model') != model or type(body.get('stream',False)) is not bool:
        raise ValueError('invalid model or stream')
    budget=body.get('max_tokens',64)
    if type(budget) is not int or not 1 <= budget <= 1024 or body.get('temperature',0) != 0:
        raise ValueError('greedy generation requires temperature zero and 1–1024 output tokens')
    messages=body.get('messages')
    if not isinstance(messages,list) or not 1 <= len(messages) <= 32:
        raise ValueError('one to 32 messages required')
    flat=[]; lengths=[]
    for message in messages:
        if not isinstance(message,dict) or set(message)!={'role','content'} or message['role'] not in ('system','user','assistant'):
            raise ValueError('unsupported message')
        content=message['content']
        parts=[{'type':'text','text':content}] if isinstance(content,str) else content
        if not isinstance(parts,list): raise ValueError('invalid content')
        if message['role'] != 'user' and any(p.get('type') in ('image_url','input_image') for p in parts if isinstance(p,dict)):
            raise ValueError('images must belong to user messages')
        lengths.append(len(parts));flat.extend(parts)
    parsed=prepare_vision_input(flat)
    images=[p for p in parsed.parts if isinstance(p,ImagePart)]
    if len(images)!=1: raise ValueError('exactly one image required in supplied history')
    normalized=[];offset=0
    for message,n in zip(messages,lengths):
        content=[]
        for part in parsed.parts[offset:offset+n]:
            if isinstance(part,TextPart):
                if marker in part.text: raise ValueError('reserved image placeholder in text')
                content.append({'type':'text','text':part.text})
            else: content.append({'type':'image'})
        normalized.append({'role':message['role'],'content':content});offset+=n
    return normalized,images[0],budget


class Worker:
    def __init__(self, profile, threads, memory, command=None):
        self.command=command or [sys.executable,'-m','server.vision_worker',str(profile)]
        self.threads=threads;self.memory=memory
        self.busy=False;self.process=None;self.request_id=None;self.cancelled=False

    async def stop(self):
        process=self.process
        if process is not None and process.returncode is None:
            try: __import__('os').killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: await asyncio.wait_for(process.wait(),2)
            except asyncio.TimeoutError:
                try: __import__('os').killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                await process.wait()

    async def events(self,payload):
        payload=dict(payload,threads=self.threads,address_space_bytes=self.memory)
        if self.cancelled: raise RuntimeError("request cancelled")
        data=json.dumps(payload,ensure_ascii=False).encode()
        if len(data)>256*1024: raise ValueError('worker input too large')
        try:
            self.process=await asyncio.create_subprocess_exec(*self.command,stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,start_new_session=True,limit=128*1024)
            if self.cancelled: raise RuntimeError("request cancelled")
            self.process.stdin.write(data);await self.process.stdin.drain();self.process.stdin.close()
            terminal=None;event_bytes=0
            while line:=await self.process.stdout.readline():
                event_bytes+=len(line)
                if event_bytes>8*1024*1024: raise ValueError("worker output exceeds budget")
                event=json.loads(line)
                if terminal is not None or not isinstance(event,dict) or event.get('type') not in ('stage','delta','complete','error'):
                    raise ValueError('invalid worker protocol')
                if event['type'] in ('complete','error'): terminal=event
                else: yield event
            status=await self.process.wait()
            if terminal is None or (status and terminal['type']=='complete'):
                raise ValueError('worker exited without successful terminal event')
            yield terminal
        except Exception as error:
            raise RuntimeError("vision worker protocol failed") from error
        finally:
            with anyio.CancelScope(shield=True):
                await self.stop();self.process=None


def create_app(profile, *, threads=4, memory=32<<30, command=None):
    from server.vision_bundle import load
    pinned=load(profile)
    worker=Worker(profile,threads,memory,command)
    readiness={}
    @asynccontextmanager
    async def lifespan(app):
        async with asyncio.timeout(300):
            async for event in worker.events({'probe':True}):
                if event['type']=='complete':
                    readiness.update(event)
                    readiness['server_instance_id']=uuid.uuid4().hex
                elif event['type']=='error': raise RuntimeError('vision preflight failed')
        if not readiness: raise RuntimeError('missing preflight result')
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await worker.stop()
    app=FastAPI(lifespan=lifespan)
    def error(code,message,status):return JSONResponse({'error':{'code':code,'message':message}},status_code=status)
    @app.get('/health')
    async def health():return {'ready':bool(readiness),'busy':worker.busy,'resident':False}
    @app.get('/v1/models')
    async def models():return {'object':'list','data':[{'id':pinned['model'],'object':'model'}]}
    @app.get('/v1/capabilities')
    async def capabilities():return readiness
    @app.delete('/v1/requests/{request_id}')
    async def cancel(request_id:str):
        if worker.request_id!=request_id:return error('not_found','No active request with this ID',404)
        worker.cancelled=True
        await worker.stop()
        return {'request_id':request_id,'status':'cancellation_requested'}
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        if worker.busy:return error('busy','One active request supported; retry after completion',429)
        worker.cancelled=False
        worker.busy=True;worker.request_id='vision-'+uuid.uuid4().hex
        rid=worker.request_id;directory=None;transferred=False
        async def release():
            with anyio.CancelScope(shield=True):
                await worker.stop()
                if directory:directory.cleanup()
                worker.busy=False;worker.request_id=None
        try:
            raw=bytearray()
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(raw)+len(chunk)>12*1024*1024:return error('body_limit','Upload exceeds 12 MiB',413)
                    raw.extend(chunk)
            body=json.loads(raw)
            messages,image,budget=validate(body,pinned['model'],pinned['image_placeholder'])
            directory=tempfile.TemporaryDirectory(prefix='cke-vision-')
            path=Path(directory.name)/('image.png' if image.mime_type=='image/png' else 'image.jpg')
            path.write_bytes(image.data)
            payload={'messages':messages,'image_path':str(path),'max_tokens':budget}
            def completion(text,event):
                return {'id':rid,'object':'chat.completion','model':pinned['model'],
                    'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':event['finish_reason']}],
                    'usage':event['usage'],'cke_evidence':event.get('evidence')}
            async def stream():
                try:
                    async for event in worker.events(payload):
                        if event['type']=='stage':yield ': native stage\n\n';continue
                        if event['type']=='delta':value={'id':rid,'object':'chat.completion.chunk','model':pinned['model'],'choices':[{'index':0,'delta':{'content':event['text']},'finish_reason':None}]}
                        elif event['type']=='complete':value={'id':rid,'object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':event['finish_reason']}],'usage':event['usage'],'cke_evidence':event.get('evidence')}
                        else:value={'error':event}
                        yield 'data: '+json.dumps(value)+'\n\n'
                    yield 'data: [DONE]\n\n'
                except Exception:
                    yield 'data: {"error":{"code":"worker_failed"}}\n\n'
                finally:await release()
            if body.get('stream',False):
                transferred=True
                return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Request-ID':rid})
            async def collect():
                text=''
                async for event in worker.events(payload):
                    if event['type']=='delta':text+=event['text']
                    elif event['type']=='error':return error(event['code'],event['message'],400 if event['code']=='context_length_exceeded' else 500)
                    elif event['type']=='complete':return JSONResponse(completion(text,event),headers={'X-Request-ID':rid})
            task=asyncio.create_task(collect())
            try:
                while not task.done():
                    await asyncio.wait({task},timeout=.1)
                    if await request.is_disconnected():
                        task.cancel();await worker.stop()
                        return error('disconnected','Client disconnected',499)
                return await task
            finally:
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
        except (ValueError,VisionInputError):return error('invalid_request','Invalid or unsupported vision request',400)
        except TimeoutError:return error('upload_timeout','Upload timed out',408)
        except Exception:return error('worker_failed','Vision worker failed; state discarded',500)
        finally:
            if not transferred:await release()
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True);parser.add_argument('--port',type=int,default=8120)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    import uvicorn
    uvicorn.run(create_app(args.profile,threads=args.threads),host='127.0.0.1',port=args.port)

if __name__=='__main__':main()
