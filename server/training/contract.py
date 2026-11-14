"""Side-effect-free v8 training plan validation; trusted local fixture registry.

JSON-lines CLI is an experimental local protocol, not an authenticated endpoint.
A validated plan never submits work, reserves resources, or executes training.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='cke.training.v1'
MAX_BYTES=16384
SHAPE=dict(layers=4,d_model=32,hidden=64,heads=4,kv_heads=4,vocab_size=256,seq_len=32)
SPECS={'svg_tiny_v8':'svg_single_document_v1.json','english_tiny_v8':'english_byte_v1.json'}

def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(encode(x)).hexdigest()
def file_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ident(x):return type(x) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',x) is not None

def decode(raw):
    if len(raw)>MAX_BYTES:raise ValueError('MESSAGE_TOO_LARGE')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('INVALID_JSON')
            d[k]=v
        return d
    def constant(_):raise ValueError('INVALID_JSON')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (ValueError,RecursionError,UnicodeError):raise ValueError('INVALID_JSON') from None

def revision(root):
    git=root/'.git'
    if git.is_file():git=(root/git.read_text().strip().removeprefix('gitdir: ')).resolve()
    head=(git/'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref=head[5:];common=(git/(git/'commondir').read_text().strip()).resolve() if (git/'commondir').exists() else git
        if (common/ref).exists():head=(common/ref).read_text().strip()
        else:head=next(line.split()[0] for line in (common/'packed-refs').read_text().splitlines() if line.endswith(' '+ref))
    if re.fullmatch('[0-9a-f]{40}',head) is None:raise ValueError('UNKNOWN_CKE_REVISION')
    return head

def registered_dataset(root,recipe):
    spec_path=root/'version/v8/training'/SPECS[recipe]
    spec=json.loads(spec_path.read_text());splits={};valid=(spec.get('schema')=='cke.v8.training_corpus.v1' and spec.get('tokenizer',{}).get('name')=='utf8_byte_v1' and spec.get('tokenizer',{}).get('vocab_size')==256)
    for name in ('train','validation'):
        row=spec[name];path=(root/row['path']).resolve()
        if not path.is_relative_to((root/'version/v8/training/corpus').resolve()):raise ValueError('INVALID_REGISTRY')
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        count=row['token_count'];ok=blob==row['git_blob'] and type(count) is int and 0<count<=len(data)
        try:
            text=data[:count].decode('utf-8')
            if recipe=='svg_tiny_v8':
                if len(data)!=count or '<!doctype' in text.lower() or '<!entity' in text.lower():raise ValueError('unsupported SVG')
                tree=ET.fromstring(text)
                if tree.tag!='{http://www.w3.org/2000/svg}svg':raise ValueError('invalid root')
                for element in tree.iter():
                    if element.tag.rsplit('}',1)[-1] not in {'svg','rect','circle'}:raise ValueError('unsupported element')
                    if any(k.rsplit('}',1)[-1] not in {'xmlns','width','height','viewBox','x','y','rx','cx','cy','r','fill'} or 'url(' in v.lower() or '://' in v for k,v in element.attrib.items()):raise ValueError('unsupported attribute')
        except (ValueError,ET.ParseError,TypeError):ok=False
        valid=valid and ok
        splits[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),token_count=count,documents=1)
    entry=dict(id=spec['name'],version='v1',format='svg_utf8' if recipe=='svg_tiny_v8' else 'text_utf8',tokenizer='utf8_byte_v1',splits=splits,
               spec_sha256=file_hash(spec_path),qc=dict(integrity='PASS' if valid else 'FAIL',training_quality='NOT_TESTED'),
               limitation=spec.get('validation_relationship','machinery fixture; no generalization claim'))
    entry['sha256']=digest(entry)
    return entry

class Backend:
    def __init__(self,root=ROOT,*,resources='unknown',clef_packages=None):
        if resources not in ('unknown','unavailable'):raise ValueError('no resource reservation provider')
        self.clef_packages=copy.deepcopy(clef_packages or {})
        self.root=Path(root);self.boot=uuid.uuid4().hex;self.resources=resources
    def capabilities(self):
        datasets={k:registered_dataset(self.root,k) for k in SPECS}
        recipes={k:dict(status='validation_only',workflow='v8_training_workflow',shape=copy.deepcopy(SHAPE),dtype='fp32',optimizer='adamw',method='full_tiny_decoder',tokenizer='utf8_byte_v1',dataset=datasets[k]['id']) for k in SPECS}
        body=dict(schema=SCHEMA,worker_boot_id=self.boot,cke_commit=revision(self.root),
                  runtime=dict(version='v8.workflow.source',loaded=False,workflow_sha256=file_hash(self.root/'version/v8/scripts/run_training_workflow_v8.py')),
                  execution='disabled',resources=self.resources,recipes=recipes,datasets=list(datasets.values()),
                  stages=dict(discover='supported',validate='supported',submit='unsupported',observe='unsupported',artifacts='unsupported',cancel='unsupported'),
                  methods=dict(full_tiny_decoder='validation_only',clef_head='unsupported',lora='unsupported',qlora='unsupported'),
                  limits=dict(message_bytes=MAX_BYTES,epochs=[1,10],grad_accum=[1,8],queued_jobs=0))
        if self.clef_packages:
            from .clef import packages, RECIPE
            from version.v8.decision.training_limits import limits as head_limits
            body['clef_packages']=packages(self.clef_packages)
            body['methods']['clef_head']='validation_only'
            body['recipes'][RECIPE]=dict(limits=head_limits(),status='validation_only',dtype='fp32',optimizer='adamw',method='head_only',geometry='one declared geometry per package',executor_sha256=file_hash(self.root/'version/v8/decision/train_clef_head.py'))
        body['sha256']=digest(body);return body
    def exchange(self,q):
        if len(encode(q))>MAX_BYTES:raise ValueError('MESSAGE_TOO_LARGE')
        errors=[];warnings=[]
        def error(field,code):errors.append(dict(field=field,code=code))
        if type(q) is not dict or not ident(q.get('request_id')):raise ValueError('INVALID_ENVELOPE')
        allowed={'schema','request_id','operation'}|({'worker_boot_id','capabilities_sha256','config'} if q.get('operation')=='validate' else set())
        for field in sorted(set(q)-allowed):error('/'+field,'UNKNOWN_FIELD')
        for field in sorted(allowed-set(q)):error('/'+field,'REQUIRED')
        if q.get('schema')!=SCHEMA:error('/schema','UNSUPPORTED_VERSION')
        if q.get('operation') not in ('discover','validate'):error('/operation','UNSUPPORTED_OPERATION')
        caps=self.capabilities()
        response=dict(schema=SCHEMA,request_id=q['request_id'],request_sha256=digest(q),worker_boot_id=self.boot,capabilities_sha256=caps['sha256'],ok=False,capabilities=None,plan=None,plan_sha256=None,errors=errors,warnings=warnings,submitted=False)
        if errors:return response
        if q['operation']=='discover':response.update(ok=True,capabilities=caps);return response
        if q['worker_boot_id']!=self.boot:error('/worker_boot_id','STALE_WORKER')
        if q['capabilities_sha256']!=caps['sha256']:error('/capabilities_sha256','STALE_CAPABILITIES')
        if type(q['config']) is dict and q['config'].get('recipe')=='clef_head_fp32_v1' and self.clef_packages:
            from .clef import validate
            errors.extend(validate(q['config'],caps['clef_packages']))
            if self.resources=='unavailable':error('/resources','UNAVAILABLE')
            if not errors:
                plan=dict(schema='cke.training.plan.v1',worker_boot_id=self.boot,capabilities_sha256=caps['sha256'],cke_commit=caps['cke_commit'],workflow_sha256=caps['recipes']['clef_head_fp32_v1']['executor_sha256'],config=copy.deepcopy(q['config']),submittable=False,execution='disabled')
                response.update(ok=True,plan=plan,plan_sha256=digest(plan))
                warnings.append(dict(field='/config/package_id',code='SUPPLIED_FEATURES_NOT_BACKBONE_TRAINING'))
            return response
        cfg=q['config'];required={'recipe','dataset','tokenizer','shape','dtype','optimizer','method','epochs','grad_accum'}
        if type(cfg) is not dict:error('/config','EXPECTED_OBJECT');return response
        for field in sorted(required-set(cfg)):error('/config/'+field,'REQUIRED')
        for field in sorted(set(cfg)-required):error('/config/'+field,'UNKNOWN_FIELD')
        recipe=cfg.get('recipe');known=type(recipe) is str and recipe in caps['recipes']
        if not known:error('/config/recipe','UNSUPPORTED_RECIPE')
        for field,expected in [('tokenizer','utf8_byte_v1'),('dtype','fp32'),('optimizer','adamw'),('method','full_tiny_decoder')]:
            if cfg.get(field)!=expected:error('/config/'+field,'UNSUPPORTED_VALUE')
        shape=cfg.get('shape')
        if type(shape) is not dict:error('/config/shape','EXPECTED_OBJECT')
        else:
            for k in sorted(set(shape)|set(SHAPE)):
                if k not in SHAPE or type(shape.get(k)) is not int or shape[k]!=SHAPE[k]:error('/config/shape/'+k,'UNSUPPORTED_DIMENSION')
        for field,bounds in [('epochs',(1,10)),('grad_accum',(1,8))]:
            if type(cfg.get(field)) is not int or not bounds[0]<=cfg[field]<=bounds[1]:error('/config/'+field,'OUT_OF_RANGE')
        dataset=cfg.get('dataset')
        if type(dataset) is not dict or set(dataset)!={'id','version','sha256','train_split','validation_split'}:error('/config/dataset','INVALID_REFERENCE')
        else:
            expected=next((d for d in caps['datasets'] if d['id']==dataset['id']),None)
            if expected is None:error('/config/dataset/id','UNKNOWN_DATASET')
            else:
                for k in ('version','sha256'):
                    if dataset[k]!=expected[k]:error('/config/dataset/'+k,'IDENTITY_MISMATCH')
                if expected['qc']['integrity']!='PASS':error('/config/dataset','QC_FAILED')
                if any(s['token_count']<SHAPE['seq_len'] for s in expected['splits'].values()):error('/config/dataset','INSUFFICIENT_TOKENS')
                if known and caps['recipes'][recipe]['dataset']!=dataset['id']:error('/config/dataset/id','INCOMPATIBLE_RECIPE')
                for field,value in [('train_split','train'),('validation_split','validation')]:
                    if dataset[field]!=value:error('/config/dataset/'+field,'UNSUPPORTED_SPLIT')
        if self.resources=='unavailable':error('/resources','UNAVAILABLE')
        else:warnings.append(dict(field='/resources',code='ADMISSION_NOT_IMPLEMENTED'))
        if not errors:
            plan=dict(schema='cke.training.plan.v1',worker_boot_id=self.boot,capabilities_sha256=caps['sha256'],cke_commit=caps['cke_commit'],workflow_sha256=caps['runtime']['workflow_sha256'],config=copy.deepcopy(cfg),submittable=False,execution='disabled')
            response.update(ok=True,plan=plan,plan_sha256=digest(plan))
            warnings.append(dict(field='/config/dataset',code='FIXTURE_NOT_QUALITY_CERTIFICATION'))
        return response

def main():
    backend=Backend()
    while True:
        raw=sys.stdin.buffer.readline(MAX_BYTES+1)
        if not raw:return
        try:
            if len(raw)>MAX_BYTES:raise ValueError('MESSAGE_TOO_LARGE')
            result=backend.exchange(decode(raw))
        except (ValueError,OSError,KeyError,TypeError):result=dict(schema=SCHEMA,ok=False,error='INVALID_REQUEST_OR_REGISTRY',submitted=False)
        encoded=encode(result)
        if len(encoded)>MAX_BYTES:encoded=encode(dict(schema=SCHEMA,ok=False,error='RESPONSE_TOO_LARGE',submitted=False))
        print(encoded.decode(),flush=True)
        if len(raw)>MAX_BYTES:return

if __name__=='__main__':main()
