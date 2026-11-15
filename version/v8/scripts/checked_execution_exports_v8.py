"""Validate and resolve named execution ports against ordinary compiler artifacts."""
import hashlib,json,math,re
from pathlib import Path

def digest_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def resolve_exports(graph,layout,calls,providers,exported=()):
    declaration=graph.get('execution_exports')
    if declaration is None:return None
    if declaration.get('schema')!='cke.checked.execution_exports.v1':raise ValueError('unsupported execution export schema')
    if set(declaration)-{'schema','ports','entries','provider_entries','properties'}:raise ValueError('unknown execution export field')
    buffers={b['name']:b for b in layout['memory']['activations']['buffers']}
    used={}
    for op in calls['operations']:
        provider=providers[op['call_abi']['kernel_id']]
        for arg in op['args']:
            name=arg.get('buffer_ref');label=arg.get('source','').partition(':')[2]
            if not name:continue
            for category in ('inputs','outputs'):
                for port in provider.get(category,[]):
                    if port['name']==label:
                        dtype={'f32':'fp32','float32':'fp32','int32':'i32'}.get(port['dtype'],port['dtype'])
                        used.setdefault(name,set()).add(dtype)
    ports={}
    for name,spec in declaration.get('ports',{}).items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) or set(spec)-{'buffer','shape','slice'}:
            raise ValueError('invalid named execution port')
        buffer=spec['buffer']
        types=used.get(buffer,set())
        if len(types)!=1:raise ValueError('execution port has missing or ambiguous provider dtype: '+name)
        dtype=next(iter(types))
        if buffer not in buffers or buffer not in graph['activation_buffers'] or dtype not in ('fp32','i32'):
            raise ValueError('unknown buffer or execution dtype')
        full=graph['activation_buffers'][buffer]['shape']
        if not isinstance(full,list) or any(type(v) is not int or v<1 for v in full):raise ValueError('invalid execution port shape')
        shape=list(full);offset=buffers[buffer]['abs_offset'];slice_spec=spec.get('slice')
        if slice_spec is not None:
            if set(slice_spec)!={'axis','start','length'} or slice_spec['axis']!=0:raise ValueError('only checked contiguous leading-axis slices are supported')
            start=slice_spec['start'];length=slice_spec['length']
            if type(start) is not int or type(length) is not int or start<0 or length<1 or start+length>shape[0]:raise ValueError('execution port slice exceeds tensor')
            offset+=start*math.prod(shape[1:])*4;shape[0]=length
        if spec['shape']!=shape:raise ValueError('execution port shape differs from compiled tensor: '+name)
        size=math.prod(shape)*4;physical=buffers[buffer]
        if offset<physical['abs_offset'] or offset+size>physical['abs_offset']+physical['size']:raise ValueError('execution port exceeds allocation')
        ports[name]={'buffer':buffer,'dtype':dtype,'shape':shape,'offset':offset,'elements':math.prod(shape),'bytes':size,'source_shape':full,'slice':slice_spec}
    from codegen_checked_calls_v8 import resolve_entry_segments
    segments=resolve_entry_segments(graph['native_entry'],calls['operations'])
    ranges={name:(start,end) for name,start,end in segments}
    if segments[0][0]!=graph['native_entry']['function']:
        ranges[graph['native_entry']['function']]=(0,len(calls['operations']))
    entries=dict(declaration.get('entries',{}))
    if any(value not in ranges for value in entries.values()):raise ValueError('execution entry not generated')
    entry_writes={}
    for function,(start,end) in ranges.items():
        writes=set()
        for op in calls['operations'][start:end]:
            output_names={port['name'] for port in providers[op['call_abi']['kernel_id']].get('outputs',[])}
            writes.update(arg['buffer_ref'] for arg in op['args']
                if arg.get('buffer_ref') and arg.get('source','').partition(':')[2] in output_names)
        entry_writes[function]=sorted(writes)
    selected={op['call_abi']['kernel_id'] for op in calls['operations']};provider_entries={}
    for name,spec in declaration.get('provider_entries',{}).items():
        if set(spec)!={'provider','function'} or spec['provider'] not in selected:
            raise ValueError('execution provider is not selected')
        if providers[spec['provider']]['impl']['function']!=spec['function'] or spec['function'] not in exported:
            raise ValueError('execution provider entry differs from exported ABI')
        if name in entries:raise ValueError('duplicate execution entry')
        entries[name]=spec['function']
        provider_entries[name]={'id':spec['provider'],'operation_interface':providers[spec['provider']].get('operation_interface'),
            'equivalence_group':providers[spec['provider']].get('selection',{}).get('equivalence_group')}
    return {'schema':declaration['schema'],'ports':ports,'entries':entries,
            'properties':declaration.get('properties',{}),'arena_bytes':layout['memory']['arena']['total_size'],
            'entry_writes':entry_writes,'entry_parameters':graph['native_entry']['params'],'provider_entries':provider_entries,
            'runtime_length_outputs':{name:{'parameter':parameter,'capacity':graph['runtime_lengths'][name]['capacity']}
                for name,parameter in graph['native_entry'].get('runtime_length_outputs',{}).items()}}

def load_exports(folder,loaded):
    from version.v8.scripts.checked_bundle_evidence_v8 import IDENTITY_SYMBOL
    import ctypes
    folder=Path(folder);document=json.loads((folder/'execution-bindings.json').read_text())
    identity=json.loads((folder/'build-identity.json').read_text());inputs=identity['identity_inputs']
    if canonical_hash(document)!=inputs.get('execution_exports_sha256'):raise ValueError('execution bindings differ from loaded build')
    for filename,key in [('call.json','call_ir_sha256'),('layout.json','layout_sha256'),('circuit.json','circuit_sha256')]:
        if digest_file(folder/filename)!=inputs.get(key):raise ValueError('stale execution artifact: '+filename)
    identity_fn=getattr(loaded,IDENTITY_SYMBOL);identity_fn.argtypes=[];identity_fn.restype=ctypes.c_char_p
    if identity_fn().decode()!=identity['build_identity_sha256']:raise ValueError('execution bindings belong to another loaded library')
    if canonical_hash(inputs)!=identity['build_identity_sha256']:raise ValueError('invalid execution identity manifest')
    document['loaded_build_identity']=identity['build_identity_sha256']
    return document
