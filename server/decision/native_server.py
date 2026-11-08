"""Loopback-only experimental generated-head evaluation worker.

No authentication/remote transport, text backbone, training, or business actions.
A timed-out client does not preempt native C; its slot remains occupied until C
returns. The independent consumer also enforces its absolute publication deadline.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from . import contract as wire
from .native import NativeHead

class NativeServer(ThreadingHTTPServer):
    daemon_threads=False
    block_on_close=True
    def __init__(self,head,port=0):
        self.head=head;self.connections=threading.BoundedSemaphore(4)
        super().__init__(('127.0.0.1',port),Handler)
    def process_request(self,request,client_address):
        if not self.connections.acquire(False):
            self.shutdown_request(request);return
        try:super().process_request(request,client_address)
        except BaseException:self.connections.release();raise
    def process_request_thread(self,request,client_address):
        try:super().process_request_thread(request,client_address)
        finally:self.connections.release()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def setup(self):
        super().setup();self.connection.settimeout(1)
        # Absolute read/response deadline, not a resettable per-byte timeout.
        self.timer=threading.Timer(2,self.abort_connection);self.timer.daemon=True;self.timer.start()
    def abort_connection(self):
        import socket
        try:self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:pass
    def finish(self):
        self.timer.cancel()
        try:super().finish()
        except OSError:pass
    def do_POST(self):
        try:
            wire.require(self.path=='/evaluate','UNSUPPORTED_TASK')
            wire.require(self.headers.get('Content-Type')=='application/json')
            wire.require(not self.headers.get('Transfer-Encoding') and len(self.headers.get_all('Content-Length',[]))==1)
            n=int(self.headers.get('Content-Length','0'));wire.require(1<=n<=wire.MAX_BYTES)
            raw=self.rfile.read(n);wire.require(len(raw)==n)
            q=wire.decode(raw);wire.validate_request(q)
            status,response=self.server.head.evaluate(q)
            body=wire.encode(response)
        except (wire.ContractError,ValueError,OSError):
            status=400;body=b'{"error":{"code":"INVALID_REQUEST"}}'
        except Exception:
            status=500;body=b'{"error":{"code":"INTERNAL_ERROR"}}'
        try:
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except OSError:pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--package-sha256',required=True);p.add_argument('--port',type=int,default=0)
    args=p.parse_args()
    head=NativeHead(args.package,args.package_sha256)
    try:
        with NativeServer(head,args.port) as server:
            print(wire.encode(dict(url=f'http://127.0.0.1:{server.server_port}/evaluate',implementation_stage='head_only',execution_kind='native_generated',package_sha256=head.package_sha256,requests=[head.request(k) for k in head.records])).decode(),flush=True)
            try:server.serve_forever()
            except KeyboardInterrupt:pass
    finally:head.close()

if __name__=='__main__':main()
