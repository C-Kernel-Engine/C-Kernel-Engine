"""Bounded decision-evaluation v1 proposal. No model executor or actions."""
import hashlib
import json
import math
import re

SCHEMA = 'cke.decision.evaluate.v1'
MAX_BYTES = 16384
MAX_CANDIDATES = 16
MAX_TIMEOUT_MS = 1000
IDENT = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z')
HASH = re.compile(r'[a-f0-9]{64}\Z')
BOOT = re.compile(r'[a-f0-9]{32}\Z')
REV = re.compile(r'[a-f0-9]{40,64}\Z')

class ContractError(ValueError):
    """Fixed code only: no private input, paths or peer messages."""

def require(ok, code='INVALID_REQUEST'):
    if not ok: raise ContractError(code)

def exact(obj, keys):
    require(type(obj) is dict and set(obj) == set(keys))

def matches(pattern, value):
    return type(value) is str and pattern.fullmatch(value) is not None

def finite(value):
    try:return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:return False

def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')

def digest(obj):
    return hashlib.sha256(encode(obj)).hexdigest()

def decode(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'MESSAGE_TOO_LARGE')
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out, 'MALFORMED_JSON');out[key]=value
        return out
    def constant(_): raise ContractError('MALFORMED_JSON')
    try:return json.loads(raw.decode('utf8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):raise ContractError('MALFORMED_JSON') from None

def validate_request(q):
    exact(q, ('schema','request_id','adapter_boot_id','worker_boot_id','model','task','input','candidate_ids','timeout_ms','policy'))
    require(q['schema']==SCHEMA and matches(IDENT,q['request_id']))
    require(matches(BOOT,q['adapter_boot_id']) and matches(BOOT,q['worker_boot_id']))
    exact(q['model'],('id','revision'))
    require(matches(IDENT,q['model']['id']) and matches(REV,q['model']['revision']))
    require(q['task']=='registered_feature_scores','UNSUPPORTED_TASK')
    exact(q['input'],('artifact_id','sha256'))
    require(matches(IDENT,q['input']['artifact_id']) and matches(HASH,q['input']['sha256']))
    ids=q['candidate_ids']
    require(type(ids) is list and 1<=len(ids)<=MAX_CANDIDATES and all(matches(IDENT,x) for x in ids))
    require(len(ids)==len(set(ids)))
    require(type(q['timeout_ms']) is int and 1<=q['timeout_ms']<=MAX_TIMEOUT_MS)
    exact(q['policy'],('minimum_logit_margin',))
    require(finite(q['policy']['minimum_logit_margin']) and 0<=q['policy']['minimum_logit_margin']<=100)
    require(len(encode(q))<=MAX_BYTES,'MESSAGE_TOO_LARGE')
    return q

def rank(ids, logits, margin):
    require(len(ids)==len(logits) and all(finite(x) for x in logits),'NONFINITE_SCORE')
    ordered=sorted(range(len(ids)),key=lambda i:(-logits[i],i))
    if len(ordered)>1:
        gap=logits[ordered[0]]-logits[ordered[1]]
        if gap==0:return dict(status='abstained',reason='tie',candidate_id=None)
        if gap<margin:return dict(status='abstained',reason='policy_margin',candidate_id=None)
    return dict(status='ranked',reason=None,candidate_id=ids[ordered[0]])
