"""Resident checked valid lengths for the connected FP32 Ming audio decoder."""
from version.v8.tts.build_ming3_audio_decoder_circuit import build_circuit as fixed
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.scripts.checked_circuit_authoring_v8 import runtime_extent as _extent, bind_runtime_scalars as _bind

def build_circuit(capacity=64,**kwargs):
    g=fixed(frames=capacity,**kwargs);c=g['contract']['runtime_invariants']
    width=c['width'];factor=c['factor'];inner=c['inner']
    g['name']+='__runtime';g['native_entry']['function']='ck_ming3_audio_decoder_runtime_fp32'
    c.update(runtime_latent_frames=True,latent_capacity=capacity)
    g['runtime_constants']={'value_elements':1,'phoneme_count':1,'max_duration':capacity,'expanded_capacity':capacity}
    limits={'latent_frames':capacity,'spectral_frames':capacity*factor,'mlp_elements':capacity*factor*inner}
    g['runtime_lengths']={k:{'producer':'check_'+k,'result':'valid_extent','capacity':v,'allow_zero':False} for k,v in limits.items()}
    for key in ['request_frames',*limits]:
        g['activation_buffers'][key]={'shape':[1]};g['activation_bindings'][key]=key
    g['block_types']['block']['header']=[
        _extent('check_latent_frames','runtime_extent_sum','request_frames','latent_frames','latent_frames',capacity=capacity),
        _extent('check_spectral_frames','runtime_extent_scale','latent_frames','spectral_frames','spectral_frames',factor=factor,capacity=capacity*factor),
        _extent('check_mlp_elements','runtime_extent_scale','spectral_frames','mlp_elements','mlp_elements',factor=inner,capacity=capacity*factor*inner)]
    for op in g['block_types']['block']['body']['ops']:
        kind=op['op']
        if kind.startswith('linear_rows'):b={'linear_rows':'latent_frames' if op['id']=='projection' else 'spectral_frames'}
        elif kind=='audio_upsample_linear_checked':b={'linear_frames':'latent_frames'}
        elif kind=='rmsnorm_rows_checked':b={'rows':'spectral_frames'}
        elif kind in ('token_head_layout_checked','rope_qk_bounded','attention_sliding_bounded'):b={'tokens':'spectral_frames'}
        elif kind=='audio_scaled_sum_strided_checked':b={'sum_rows':'spectral_frames'}
        elif kind=='silu_mul_rows_checked':b={'elements':'mlp_elements'}
        elif kind=='audio_spectrum_cartesian_checked':b={'spectrum_frames':'spectral_frames'}
        elif kind=='audio_istft_complex_window_checked':b={'istft_frames':'spectral_frames'}
        else:raise ValueError('unbound decoder consumer '+kind)
        _bind(op,**b)
    # Return the runtime's checked valid length, rather than host reconstruction.
    crop=(c['n_fft']-c['hop'])//2;extra=c['n_fft']-c['hop']-2*crop
    g['runtime_lengths']['valid_samples']={'producer':'check_valid_samples','result':'valid_extent',
        'capacity':c['samples'],'allow_zero':False}
    g['activation_buffers']['valid_samples']={'shape':[1]};g['activation_bindings']['valid_samples']='valid_samples'
    g['block_types']['block']['header'].append(_extent('check_valid_samples','runtime_extent_affine',
        'spectral_frames','valid_samples','valid_samples',factor=c['hop'],offset=extra,capacity=c['samples']))
    g['native_entry']['params'].append({'c_type':'int32_t *','name':'valid_samples_out'})
    g['native_entry']['runtime_length_outputs']={'valid_samples':'valid_samples_out'}
    g['execution_exports']={'schema':'cke.checked.execution_exports.v1',
        'entries':{'run':g['native_entry']['function']},
        'ports':{name:execution_port(g,buffer) for name,buffer,dtype in (
            ('latent','latent','fp32'),('request_frames','request_frames','i32'),('window','window','fp32'),('waveform','output','fp32'))},
        'properties':{'sample_rate_hz':44100,'window':'periodic_hann_f32',
            'valid_samples_output':'valid_samples_out','valid_samples_length':'valid_samples'}}
    return g
