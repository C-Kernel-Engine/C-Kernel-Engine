"""Resident checked valid lengths for the connected FP32 Ming audio decoder."""
from version.v8.tts.build_ming3_audio_decoder_circuit import build_circuit as fixed
from version.v8.tts.build_qwen3_tts_runtime_prompt_first_block_circuit import _extent,_bind

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
    return g
