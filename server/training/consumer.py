"""Independent simulated external validation client. No producer imports/jobs."""
import hashlib
import json
import math
import re
import threading
import time

class InvalidReply(ValueError):pass

def check(ok):
    if not ok:raise InvalidReply('INVALID_OR_STALE_REPLY')
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def hexid(x,n):return type(x) is str and re.fullmatch('[a-f0-9]{'+str(n)+'}',x) is not None

def validate(reply,request,capabilities=None):
    check(type(request) is dict and request.get('schema')=='cke.training.v1' and type(request.get('request_id')) is str)
    check(request.get('operation') in ('discover','validate'))
    check(type(reply) is dict and set(reply)=={'schema','request_id','request_sha256','worker_boot_id','capabilities_sha256','ok','capabilities','plan','plan_sha256','errors','warnings','submitted'})
    check(reply['schema']=='cke.training.v1' and reply['request_id']==request['request_id'] and reply['request_sha256']==sha(request))
    check(hexid(reply['worker_boot_id'],32) and hexid(reply['capabilities_sha256'],64) and type(reply['ok']) is bool and reply['submitted'] is False)
    for name in ('errors','warnings'):
        check(type(reply[name]) is list and len(reply[name])<=128)
        for item in reply[name]:check(type(item) is dict and set(item)=={'field','code'} and type(item['field']) is str and item['field'].startswith('/') and type(item['code']) is str and bool(item['code']))
    check(reply['ok']==(not reply['errors']))
    if not reply['ok']:
        check(reply['capabilities'] is None and reply['plan'] is None and reply['plan_sha256'] is None);return reply
    if request['operation']=='discover':
        caps=reply['capabilities'];check(type(caps) is dict)
        has_head='clef_packages' in caps
        check(set(caps)-({'clef_packages'} if has_head else set())=={'schema','worker_boot_id','cke_commit','runtime','execution','resources','recipes','datasets','stages','methods','limits','sha256'})
        check(caps['schema']=='cke.training.v1' and caps['worker_boot_id']==reply['worker_boot_id'] and caps['sha256']==reply['capabilities_sha256'])
        check(caps['sha256']==sha({k:v for k,v in caps.items() if k!='sha256'}))
        check(hexid(caps['cke_commit'],40) and caps['execution']=='disabled' and caps['resources'] in ('unknown','unavailable'))
        check(caps['stages']==dict(discover='supported',validate='supported',submit='unsupported',observe='unsupported',artifacts='unsupported',cancel='unsupported'))
        check(type(caps['runtime']) is dict and set(caps['runtime'])=={'version','loaded','workflow_sha256'} and caps['runtime']['version']=='v8.workflow.source')
        check(caps['runtime'].get('loaded') is False and hexid(caps['runtime'].get('workflow_sha256'),64))
        check(type(caps['datasets']) is list and len(caps['datasets'])==2 and type(caps['recipes']) is dict and set(caps['recipes'])==({'svg_tiny_v8','english_tiny_v8','clef_head_fp32_v1'} if has_head else {'svg_tiny_v8','english_tiny_v8'}))
        check(caps['methods']==dict(full_tiny_decoder='validation_only',clef_head='validation_only' if has_head else 'unsupported',lora='unsupported',qlora='unsupported'))
        check(caps['limits']==dict(message_bytes=16384,epochs=[1,10],grad_accum=[1,8],queued_jobs=0))
        if has_head:
            packages=caps['clef_packages']
            check(type(packages) is dict and 1<=len(packages)<=8)
            for name,p in packages.items():
                check(type(p) is dict and set(p)=={'id','sha256','model','geometry','splits','feature_provenance'})
                check(p['id']==name and hexid(p['sha256'],64) and p['feature_provenance']=='supplied')
                check(type(p['splits']) is dict and set(p['splits'])=={'train','validation'} and all(type(n) is int and 1<=n<=4095 for n in p['splits'].values()))
                check(type(p['model']) is dict and type(p['geometry']) is dict)
            r=caps['recipes']['clef_head_fp32_v1']
            check(type(r) is dict and set(r)=={'limits','status','dtype','optimizer','method','geometry','executor_sha256'})
            check(r['limits']==dict(epochs=[1,1000],grad_accum=1,records=4096,tokens=512,steps=100000,dataset_bytes=536870912,lengths="exact_geometry; padding and mixed lengths unsupported") and r['status']=='validation_only' and r['dtype']=='fp32' and r['optimizer']=='adamw' and r['method']=='head_only' and r['geometry']=='one declared geometry per package' and hexid(r['executor_sha256'],64))
        for name,recipe in caps['recipes'].items():
            if name=='clef_head_fp32_v1':continue
            check(type(recipe) is dict and set(recipe)=={'status','workflow','shape','dtype','optimizer','method','tokenizer','dataset'})
            check(recipe['status']=='validation_only' and recipe['workflow']=='v8_training_workflow' and recipe['dtype']=='fp32' and recipe['optimizer']=='adamw' and recipe['method']=='full_tiny_decoder' and recipe['tokenizer']=='utf8_byte_v1')
            check(recipe['shape']==dict(layers=4,d_model=32,hidden=64,heads=4,kv_heads=4,vocab_size=256,seq_len=32))
        for d in caps['datasets']:
            check(type(d) is dict and set(d)=={'id','version','format','tokenizer','splits','spec_sha256','qc','limitation','sha256'})
            check(type(d['id']) is str and bool(re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',d['id'])) and d['version']=='v1' and d['tokenizer']=='utf8_byte_v1' and d['format'] in ('svg_utf8','text_utf8'))
            check(hexid(d['spec_sha256'],64) and set(d['splits'])=={'train','validation'} and d['qc']['integrity'] in ('PASS','FAIL') and d['qc']['training_quality']=='NOT_TESTED')
            for split in d['splits'].values():
                check(set(split)=={'sha256','bytes','token_count','documents'} and hexid(split['sha256'],64))
                check(all(type(split[k]) is int and split[k]>0 for k in ('bytes','token_count','documents')) and split['token_count']<=split['bytes'])
            check(type(d) is dict and hexid(d.get('sha256'),64) and d['sha256']==sha({k:v for k,v in d.items() if k!='sha256'}))
        check(reply['plan'] is None and reply['plan_sha256'] is None)
    else:
        check(hexid(request.get('worker_boot_id'),32) and hexid(request.get('capabilities_sha256'),64) and type(request.get('config')) is dict)
        check(type(capabilities) is dict and capabilities.get('sha256')==request['capabilities_sha256'])
        check(capabilities['sha256']==sha({k:v for k,v in capabilities.items() if k!='sha256'}))
        check(reply['worker_boot_id']==request['worker_boot_id'] and reply['capabilities_sha256']==request['capabilities_sha256'])
        cfg=request['config']
        head=cfg.get('recipe')=='clef_head_fp32_v1'
        if head:
            check(set(cfg)=={'recipe','package_id','package_sha256','dtype','optimizer','method','epochs','learning_rate'})
            check(type(cfg['package_id']) is str and cfg['package_id'] in capabilities.get('clef_packages',{}))
            check(cfg['package_sha256']==capabilities['clef_packages'][cfg['package_id']]['sha256'])
            check(cfg['dtype']=='fp32' and cfg['optimizer']=='adamw' and cfg['method']=='head_only')
            check(type(cfg['epochs']) is int and 1<=cfg['epochs']<=1000)
            check(cfg['epochs']*capabilities['clef_packages'][cfg['package_id']]['splits']['train']<=100000)
            check(type(cfg['learning_rate']) in (int,float) and math.isfinite(cfg['learning_rate']) and 0<cfg['learning_rate']<=.01)
            check(capabilities['resources']=='unknown')
        else:
            check(set(cfg)=={'recipe','dataset','tokenizer','shape','dtype','optimizer','method','epochs','grad_accum'})
            check(type(cfg['recipe']) is str and cfg['recipe'] in capabilities['recipes'])
            recipe=capabilities['recipes'][cfg['recipe']]
            for key in ('tokenizer','shape','dtype','optimizer','method'):check(cfg[key]==recipe[key])
            for key,limit in (('epochs',10),('grad_accum',8)):check(type(cfg[key]) is int and 1<=cfg[key]<=limit)
            dataset=cfg['dataset'];check(type(dataset) is dict and set(dataset)=={'id','version','sha256','train_split','validation_split'})
            registered=next((d for d in capabilities['datasets'] if d['id']==recipe['dataset']),None);check(registered is not None)
            check(all(dataset[k]==registered[k] for k in ('id','version','sha256')) and registered['qc']['integrity']=='PASS')
            check(dataset['train_split']=='train' and dataset['validation_split']=='validation' and capabilities['resources']=='unknown')
        plan=reply['plan'];check(type(plan) is dict and set(plan)=={'schema','worker_boot_id','capabilities_sha256','cke_commit','workflow_sha256','config','submittable','execution'})
        check(plan['schema']=='cke.training.plan.v1' and plan['worker_boot_id']==request['worker_boot_id'] and plan['capabilities_sha256']==request['capabilities_sha256'])
        check(plan['cke_commit']==capabilities['cke_commit'] and plan['workflow_sha256']==(capabilities['recipes']['clef_head_fp32_v1']['executor_sha256'] if head else capabilities['runtime']['workflow_sha256']))
        check(plan['submittable'] is False and plan['execution']=='disabled' and plan['config']==request['config'])
        check(hexid(plan['cke_commit'],40) and hexid(plan['workflow_sha256'],64) and hexid(reply['plan_sha256'],64) and reply['plan_sha256']==sha(plan))
        check(reply['capabilities'] is None)
    return reply

class Consumer:
    def __init__(self):self.lock=threading.Lock();self.generation=0;self.current=None
    def cancel(self):
        with self.lock:self.generation+=1;self.current=None
    def evaluate(self,request,exchange,timeout_ms=1000,capabilities=None):
        check(type(timeout_ms) is int and 1<=timeout_ms<=1000)
        with self.lock:self.generation+=1;generation=self.generation;self.current=None
        deadline=time.monotonic()+timeout_ms/1000
        result=validate(exchange(request),request,capabilities)
        with self.lock:
            check(generation==self.generation and time.monotonic()<deadline)
            # Rejected plans never become a current accepted plan.
            if result['ok']:self.current=result
        return result
