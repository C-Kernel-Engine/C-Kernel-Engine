"""Independent simulated Antsand job observer; no worker implementation imports."""
import math
import re

class InvalidObservation(ValueError):pass

def check(ok):
    if not ok:raise InvalidObservation('INVALID_OR_STALE_OBSERVATION')
def hexid(x,n):return type(x) is str and re.fullmatch('[a-f0-9]{'+str(n)+'}',x) is not None

def observe(value,*,worker_boot_id,owner,job_id,plan_sha256,experiment_id,draft_revision,last=None,allow_simulation=False):
    check(type(value) is dict and set(value)=={'job','worker_boot_id','observed_at','cancel_supported'})
    check(hexid(worker_boot_id,32) and value['worker_boot_id']==worker_boot_id and value['cancel_supported'] is False)
    j=value['job'];check(type(j) is dict and set(j)=={'schema','job_id','owner','experiment_id','draft_revision','submitted_worker_boot_id','cke_commit','plan_sha256','status','stage','created_at','updated_at','error_code','artifacts','metrics','execution_kind','progress'})
    check(j['schema']=='cke.training.job.v1' and hexid(job_id,32) and j['job_id']==job_id and hexid(plan_sha256,64) and j['plan_sha256']==plan_sha256)
    check(type(owner) is str and bool(owner) and j['owner']==owner and j['experiment_id']==experiment_id and type(draft_revision) is int and j['draft_revision']==draft_revision)
    check(hexid(j['submitted_worker_boot_id'],32) and hexid(j['cke_commit'],40))
    for t in (value['observed_at'],j['created_at'],j['updated_at']):check(type(t) in (float,int) and math.isfinite(t) and t>0)
    check(j['created_at']<=j['updated_at']<=value['observed_at'])
    check(j['status'] in ('queued','running','succeeded','failed','recovery_required'))
    check(j['stage']=={'queued':'admission','running':'workflow','succeeded':'complete','failed':'complete','recovery_required':'unknown'}[j['status']])
    check(j['execution_kind']=='native_generated_workflow' or (allow_simulation and j['execution_kind']=='contract_simulation'))
    check(j['progress'] is None and type(j['metrics']) is list and len(j['metrics'])<=2)
    for metric in j['metrics']:
        check(type(metric) is dict and set(metric)=={'name','step','step_kind','value','unit','observed_at'})
        check(metric['name'] in ('training_cross_entropy','validation_cross_entropy') and type(metric['step']) is int and metric['step']>=2 and metric['step_kind']=='epoch' and metric['unit']=='nats/token')
        check(type(metric['value']) in (int,float) and math.isfinite(metric['value']) and metric['value']>=0)
        check(type(metric['observed_at']) in (int,float) and j['created_at']<=metric['observed_at']<=value['observed_at'])
    check(type(j['artifacts']) is list and len(j['artifacts'])<=16)
    ids=[]
    for a in j['artifacts']:
        check(type(a) is dict and set(a)=={'artifact_id','type','bytes','sha256'})
        check(hexid(a['artifact_id'],32) and hexid(a['sha256'],64) and type(a['bytes']) is int and 0<a['bytes']<=16*1024**2)
        check(a['type'] in ('execution_plan','training_report','checkpoint_manifest','training_plan','training_pipeline'));ids.append(a['artifact_id'])
    check(len(ids)==len(set(ids)))
    if j['status']=='succeeded':check(j['error_code'] is None and {'training_report','execution_plan'}<={a['type'] for a in j['artifacts']})
    elif j['status'] in ('failed','recovery_required'):check(type(j['error_code']) is str and bool(j['error_code']))
    else:check(j['error_code'] is None and not j['artifacts'])
    if last is not None:
        check(last['job']['job_id']==job_id and last['job']['plan_sha256']==plan_sha256)
        check(value['observed_at']>=last['observed_at'] and j['updated_at']>=last['job']['updated_at'])
        if last['job']['status'] in ('succeeded','failed','recovery_required'):check(j==last['job'])
        if j['updated_at']==last['job']['updated_at']:check(j==last['job'])
    return value

def verify_artifacts(observation,fetch):
    """Independent bounded retrieval check; fetch accepts opaque ID and offset."""
    import base64
    import hashlib
    verified=[]
    for artifact in observation['job']['artifacts']:
        offset=0;h=hashlib.sha256()
        while True:
            chunk=fetch(artifact['artifact_id'],offset)
            check(type(chunk) is dict and set(chunk)=={'artifact_id','sha256','bytes','offset','next_offset','eof','base64'})
            check(chunk['artifact_id']==artifact['artifact_id'] and chunk['sha256']==artifact['sha256'] and chunk['bytes']==artifact['bytes'] and chunk['offset']==offset)
            try:raw=base64.b64decode(chunk['base64'],validate=True)
            except (ValueError,TypeError):raise InvalidObservation('INVALID_ARTIFACT') from None
            check(0<len(raw)<=8192 and chunk['next_offset']==offset+len(raw) and offset+len(raw)<=artifact['bytes'])
            h.update(raw);offset+=len(raw)
            check(type(chunk['eof']) is bool and chunk['eof']==(offset==artifact['bytes']))
            if chunk['eof']:break
        check(h.hexdigest()==artifact['sha256']);verified.append(artifact['artifact_id'])
    return verified
