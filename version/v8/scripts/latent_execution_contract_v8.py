"""Resolve one declared bounded latent-autoregressive execution pattern.

This is a versioned runtime-shell contract, not an arbitrary graph executor.
Model-specific profiles select named compiler exports; no model names are tested.
"""
import copy,math
import numpy as np
from version.v8.scripts.checked_execution_exports_v8 import canonical_hash

POLICY={'hidden_selection':'last_valid_row','stop':'two_logits_stop1_gt_continue0',
        'feedback':'append_one_embedding','history':'replace_one_patch','publication':'normal_stop_only',
        'aggregator_class_id':0,'rng_stream':'base_plus_frame'}

SHELLS={'checked_latent_loop.v1':{'entry':'ck_run_checked_latent_loop','policy_version':1,
    'sources':['src/runtime/checked_guided_flow.c','src/runtime/checked_latent_loop.c']}}

def native_shell_spec(descriptor):
    if descriptor.get('schema')!='cke.latent_autoregressive.v1' or descriptor.get('shell') not in SHELLS:
        raise ValueError('unsupported latent execution shell')
    if descriptor.get('policy')!=POLICY:raise ValueError('unsupported latent loop policy')
    return copy.deepcopy(SHELLS[descriptor['shell']])

def resolve_latent_execution(descriptor,exports):
    if descriptor.get('schema')!='cke.latent_autoregressive.v1' or descriptor.get('shell')!='checked_latent_loop.v1':
        raise ValueError('unsupported latent execution shell')
    if descriptor.get('policy')!=POLICY:raise ValueError('unsupported latent loop policy')
    if set(descriptor)!={'schema','shell','policy','components','sampling','minimum_frames','prediction_selection'}:
        raise ValueError('unknown latent execution field')
    if descriptor['prediction_selection']!='last_leading_row':raise ValueError('unsupported prediction selection')
    shell=native_shell_spec(descriptor)
    components={}
    required={'backbone':('embeddings','hidden','stop_logits','request_rows','token_ids'),
        'flow_model':('latent','condition','history','time','prediction'),
        'flow_update':('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output'),
        'aggregator':('latent','class_id','embedding'),'decoder':('latent','request_frames','window','waveform')}
    if set(descriptor['components'])!=set(required):raise ValueError('unexpected execution component')
    for role,roles in required.items():
        declaration=descriptor['components'][role]
        if set(declaration)!={'artifact','ports','entries'}:raise ValueError('unknown component binding field')
        artifact=exports[declaration['artifact']]
        if artifact['schema']!='cke.checked.execution_exports.v1':raise ValueError('invalid compiled execution exports')
        ports={name:artifact['ports'][symbol] for name,symbol in declaration['ports'].items()}
        entries={name:artifact['entries'][symbol] for name,symbol in declaration['entries'].items()}
        if not set(roles)<=ports.keys():raise ValueError('missing execution role: '+role)
        components[role]={'artifact':declaration['artifact'],'ports':ports,'entries':entries,'exports':artifact}
    entry_outputs={'backbone':{'embed':('embeddings',),'prefill':('hidden','stop_logits')},
        'flow_model':{'run':('prediction',)},'flow_update':{'update':('output',)},
        'aggregator':{'run':('embedding',)},'decoder':{'run':('waveform',)}}
    if 'decode' in components['backbone']['entries']:
        entry_outputs['backbone']['decode']=('decode_hidden','decode_stop_logits')
    for role,entry_roles in entry_outputs.items():
        component=components[role]
        expected_entries=set(entry_roles)|({'noise'} if role=='flow_update' else set())
        if set(component['entries'])!=expected_entries:raise ValueError('unexpected native entry roles: '+role)
        expected_ports=set(required[role])
        if role=='backbone' and 'decode' in entry_roles:
            expected_ports.update(('decode_input','decode_hidden','decode_stop_logits','decode_position'))
        if set(component['ports'])!=expected_ports:raise ValueError('unexpected native port roles: '+role)
        for entry,outputs in entry_roles.items():
            writes=component['exports'].get('entry_writes',{}).get(component['entries'][entry],[])
            if any(component['ports'][name]['buffer'] not in writes for name in outputs):
                raise ValueError('native entry does not produce its bound outputs: '+role+'.'+entry)
    def shape(role,name,expected,dtype='fp32'):
        p=components[role]['ports'][name]
        if p['shape']!=expected or p['dtype']!=dtype:raise ValueError('execution shape/dtype mismatch: '+role+'.'+name)
    def scalar(role,name,dtype='fp32'):shape(role,name,[1],dtype)
    b=components['backbone'];m=components['flow_model'];u=components['flow_update'];a=components['aggregator'];d=components['decoder']
    if len(b['ports']['embeddings']['shape'])!=2 or len(m['ports']['latent']['shape'])!=2:raise ValueError('invalid row geometry')
    capacity,width=b['ports']['embeddings']['shape'];patches,latent=m['ports']['latent']['shape']
    if patches!=1 or not 1<=capacity<=512 or not 1<=width<=8192 or not 1<=latent<=4096:raise ValueError('unsupported latent geometry')
    for name in ('embeddings','hidden'):shape('backbone',name,[capacity,width])
    shape('backbone','stop_logits',[capacity,2]);shape('backbone','token_ids',[capacity],'i32');scalar('backbone','request_rows','i32')
    cached='decode' in b['entries']
    if cached:
        for name in ('decode_input','decode_hidden'):shape('backbone',name,[1,width])
        shape('backbone','decode_stop_logits',[1,2]);scalar('backbone','decode_position','i32')
    for name in ('latent','history','prediction'):shape('flow_model',name,[1,latent])
    shape('flow_model','condition',[1,width]);scalar('flow_model','time')
    prediction=m['ports']['prediction'];selection=prediction['slice'] or {'start':0,'length':prediction['source_shape'][0]}
    if selection['length']!=1 or selection['start']!=prediction['source_shape'][0]-1:raise ValueError('prediction export is not the declared final row')
    for name in ('latent','prediction','null_prediction','noise','output'):shape('flow_update',name,[latent])
    for name in ('neg_one','cfg','dt','noise_amplitude'):scalar('flow_update',name)
    noise_symbol=descriptor['components']['flow_update']['entries']['noise']
    noise_contract=u['exports'].get('provider_entries',{}).get(noise_symbol,{})
    if noise_contract.get('operation_interface')!='audio.noise.counter_gaussian.fp32.v1' or noise_contract.get('equivalence_group')!='audio_noise_counter_box_muller_fp64':
        raise ValueError('noise callback ABI or numerical contract mismatch')
    shape('aggregator','latent',[1,latent]);shape('aggregator','embedding',[1,width]);scalar('aggregator','class_id','i32')
    if len(d['ports']['latent']['shape'])!=2:raise ValueError('invalid decoder latent geometry')
    max_frames,decoder_latent=d['ports']['latent']['shape']
    shape('decoder','latent',[max_frames,latent]);scalar('decoder','request_frames','i32')
    minimum=descriptor['minimum_frames']
    if type(minimum) is not int or not 1<=minimum<=max_frames<=512:raise ValueError('invalid generation bounds')
    for name in ('window','waveform'):
        port=d['ports'][name]
        if len(port['shape'])!=1 or port['dtype']!='fp32':raise ValueError('invalid waveform/window export')
    properties=d['exports']['properties'];lengths=d['exports']['runtime_length_outputs']
    if properties.get('window')!='periodic_hann_f32':raise ValueError('unsupported window contract')
    length=lengths.get(properties.get('valid_samples_length'))
    if not length or length['parameter']!=properties.get('valid_samples_output') or length['capacity']!=d['ports']['waveform']['elements']:
        raise ValueError('native sample-count output is not bound to waveform capacity')
    expected_parameters=[{'c_type':'uint8_t *','name':'arena'},{'c_type':'size_t','name':'arena_bytes'}]
    for role in ('backbone','flow_model','flow_update','aggregator'):
        if components[role]['exports']['entry_parameters']!=expected_parameters:raise ValueError('unsupported native component ABI')
    if d['exports']['entry_parameters']!=expected_parameters+[{'c_type':'int32_t *','name':length['parameter']}]:raise ValueError('unsupported decoder ABI')
    rate=properties.get('sample_rate_hz')
    if type(rate) is not int or not 8000<=rate<=192000:raise ValueError('invalid sample rate')
    sampling=descriptor['sampling']
    if set(sampling)!={'contract','schedule_id','times','cfg','sigma','temperature','rng'} or sampling['contract']!='guided_cfg_euler_sde.v1' or sampling['rng']!='splitmix64_box_muller.v1':
        raise ValueError('unsupported sampling contract')
    times=np.asarray(sampling['times'],dtype='f4')
    if times.ndim!=1 or not 2<=len(times)<=65 or not np.isfinite(times).all() or times[0]!=0 or times[-1]!=1 or not (np.diff(times)>0).all():raise ValueError('invalid sampling schedule')
    for name,low,high in [('cfg',-100,100),('sigma',0,100),('temperature',0,100)]:
        value=sampling[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:raise ValueError('invalid sampling coefficient')
    return {'schema':descriptor['schema'],'descriptor_sha256':canonical_hash(descriptor),'policy_version':shell['policy_version'],'shell_entry':shell['entry'],
        'components':copy.deepcopy(components),'width':width,'latent':latent,'capacity':capacity,'max_frames':max_frames,
        'minimum_frames':minimum,'steps':len(times)-1,'times':times,'noise_elements':len(times)*latent,
        'cfg':sampling['cfg'],'sigma':sampling['sigma'],'temperature':sampling['temperature'],
        'sample_rate':rate,'window_elements':d['ports']['window']['elements'],'cached':cached}
