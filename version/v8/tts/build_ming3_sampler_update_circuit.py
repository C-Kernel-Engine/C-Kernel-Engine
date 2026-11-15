"""Map-selected flow arithmetic and native counter RNG for the guided-flow shell."""
import copy
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.tts.build_ming3_flow_step_circuit import build_circuit as flow_step

def build_circuit(elements=64,steps=10):
    if type(steps) is not int or not 1<=steps<=64:raise ValueError('unsupported step budget')
    g=flow_step(elements);g['name']='ming3_sampler_update_fp32_bounded'
    g['contract']['runtime_invariants'].update(width=elements,heads=1,tokens=1,mult=1,steps=steps,
        native_rng=True,rng_contract='SplitMix64/Box-Muller; distinct from PyTorch')
    count=(steps+1)*elements
    g['activation_buffers']['native_noise']={'shape':[count]};g['activation_bindings']['native_noise']='native_noise'
    noise={'id':'native_noise','op':'audio_noise_counter_gaussian','kernel':'audio_noise_counter_gaussian_f32',
        'returns_status':True,'graph_slots':{'inputs':{},'outputs':{'output':'native_noise'}},
        'params':{'N':count,'seed':731,'stream':0,'call_constants':{'noise_output_capacity':count,'noise_count':count}}}
    g['block_types']['flow_step']['body']['ops'].insert(0,noise)
    g['native_entry']['function']='ck_ming3_sampler_update_fp32_bounded'
    g['native_entry']['segments']=[{'function':'ck_ming3_sampler_noise_fixture','last_op':'native_noise'},
        {'function':'ck_ming3_sampler_update','last_op':'stochastic_update'}]
    g['semantic_checkpoints']['exports']['native_noise']={'section':'body','template_op_id':'native_noise',
        'op':noise['op'],'checkpoints':[{'id':'ming3.sampler.native_noise','producer':'native_noise',
        'tensor':'native_noise','logical_layout':'contiguous','axis_names':['element'],'storage_dtype':'fp32'}]}
    g['execution_exports']={'schema':'cke.checked.execution_exports.v1',
        'entries':{'update':'ck_ming3_sampler_update'},
        'provider_entries':{'noise':{'provider':'audio_noise_counter_gaussian_f32','function':'audio_noise_counter_gaussian_f32'}},
        'ports':{name:execution_port(g,name) for name in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output')}}
    return g
