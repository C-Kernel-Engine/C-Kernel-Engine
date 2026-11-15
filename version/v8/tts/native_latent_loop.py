"""ABI for the checked native speech-latent loop (full-prefix baseline)."""
import ctypes as C
from version.v8.tts.native_guided_flow import ENTRY,U,S,P,Request

class LatentLoop(C.Structure):
    _fields_=[(k,ENTRY) for k in ('embed','prefix','aggregate')]+[
        (k,U) for k in ('backbone_arena','aggregator_arena')]+[
        (k,S) for k in ('backbone_bytes','aggregator_bytes','request_rows','embeddings','hidden','stop_logits',
                       'aggregate_latent','aggregate_class_id','aggregate_output','width','latent_width',
                       'capacity','prompt_rows','max_frames','min_frames')]+[
        ('flow',C.POINTER(Request)),('staging',P),('output',P),('staging_elements',S),
        ('output_elements',S),('valid_frames',C.POINTER(S)),('decode',ENTRY)]+[
        (k,S) for k in ('decode_input','decode_hidden','decode_stop_logits','decode_position')]+[('policy_version',C.c_uint32)]


def bind_latent_request(plan,bundles,token_ids,seed,cancel):
    """Bind only centrally resolved, build-identified exports to the native ABI."""
    import numpy as np
    from version.v8.tts.native_guided_flow import Model,Update,NOISE
    from version.v8.scripts.checked_bundle_evidence_v8 import IDENTITY_SYMBOL
    if type(seed) is not int or not 0<=seed<2**64:raise ValueError('invalid seed')
    if not token_ids or len(token_ids)+plan['minimum_frames']-1>plan['capacity'] or any(type(x) is not int or not 0<=x<2**31 for x in token_ids):
        raise ValueError('invalid or over-capacity token input')
    parts=plan['components'];bound={}
    for role,part in parts.items():
        bundle=bundles[part['artifact']]
        if len(bundle[1])!=part['exports']['arena_bytes']:raise ValueError('native arena differs from execution plan')
        identity=getattr(bundle[3],IDENTITY_SYMBOL);identity.argtypes=[];identity.restype=C.c_char_p
        if identity().decode()!=part['exports']['loaded_build_identity']:raise ValueError('native library differs from execution plan')
        bound[role]=bundle
    def entry(role,name):return ENTRY((parts[role]['entries'][name],bound[role][3]))
    def offset(role,name):return parts[role]['ports'][name]['offset']
    def arena(role):return C.cast(bound[role][1],U)
    def size(role):return len(bound[role][1])
    width=plan['width'];latent=plan['latent'];frames=plan['max_frames']
    cond=np.zeros(width,'f4');history=np.zeros(latent,'f4');times=plan['times'].copy()
    workspace=np.zeros(plan['noise_elements'],'f4');patch=np.zeros(latent,'f4')
    model=Model(entry('flow_model','run'),arena('flow_model'),size('flow_model'),
        *[offset('flow_model',name) for name in ('latent','time','condition','history','prediction')])
    update=Update(entry('flow_update','update'),arena('flow_update'),size('flow_update'),
        *[offset('flow_update',name) for name in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output')])
    noise=NOISE((parts['flow_update']['entries']['noise'],bound['flow_update'][3]))
    flow=Request(model,update,noise,cond.ctypes.data_as(P),history.ctypes.data_as(P),times.ctypes.data_as(P),
        latent,width,latent,plan['steps'],plan['cfg'],plan['sigma'],plan['temperature'],seed,0,
        workspace.ctypes.data_as(P),workspace.size,patch.ctypes.data_as(P),patch.size,cancel,None)
    staging=np.zeros((frames,latent),'f4');output=np.full_like(staging,np.nan);valid=S(0)
    request=LatentLoop(entry('backbone','embed'),entry('backbone','prefill'),entry('aggregator','run'),
        arena('backbone'),arena('aggregator'),size('backbone'),size('aggregator'),
        *[offset('backbone',name) for name in ('request_rows','embeddings','hidden','stop_logits')],
        *[offset('aggregator',name) for name in ('latent','class_id','embedding')],width,latent,
        plan['capacity'],len(token_ids),frames,plan['minimum_frames'],C.pointer(flow),
        staging.ctypes.data_as(P),output.ctypes.data_as(P),staging.size,output.size,C.pointer(valid))
    request.policy_version=plan['policy_version']
    if plan['cached']:
        request.decode=entry('backbone','decode')
        for name in ('decode_input','decode_hidden','decode_stop_logits','decode_position'):
            setattr(request,name,offset('backbone',name))
    port=parts['backbone']['ports']['token_ids']
    ids=np.ndarray(tuple(port['shape']),np.int32,buffer=bound['backbone'][1],offset=port['offset'])
    ids[:len(token_ids)]=token_ids
    keepalive=(cond,history,times,workspace,patch,staging,flow,cancel,bound)
    return request,output,valid,keepalive
