#!/usr/bin/env python3
"""Upload one PNG/JPEG to the experimental CKE vision Chat Completions service.

Standalone standard-library example; no CKE import or model files required.
"""
import argparse
import base64
import json
from pathlib import Path
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8120')
    parser.add_argument('--model',required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--prompt',default='Describe the image.')
    parser.add_argument('--max-tokens',type=int,default=32)
    parser.add_argument('--stream',action='store_true')
    args=parser.parse_args()
    with args.image.open('rb') as source:data=source.read(8*1024*1024+1)
    if len(data)>8*1024*1024:parser.error('image exceeds 8 MiB')
    if data.startswith(b'\x89PNG\r\n\x1a\n'):mime='image/png'
    elif data.startswith(b'\xff\xd8'):mime='image/jpeg'
    else:parser.error('only PNG/JPEG accepted; render PDF pages in the client')
    body={'model':args.model,'max_tokens':args.max_tokens,'stream':args.stream,'messages':[
        {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode()}},
                                  {'type':'text','text':args.prompt}]}]}
    request=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:
        for line in response:print(line.decode().rstrip(),flush=True)

if __name__=='__main__':main()
