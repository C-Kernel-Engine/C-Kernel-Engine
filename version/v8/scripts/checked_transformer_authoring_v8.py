"""Shared checked RMSNorm/GQA/RoPE/SwiGLU transformer circuit composition.

This authors tensor edges and selected operation contracts. It does not lower,
compile or execute them. Inputs, weight namespace and attention policy are explicit.
"""
from version.v8.scripts.checked_circuit_authoring_v8 import linear

def transformer_stack(*,tokens,width,heads,kv_heads,depth,inner,source,
                      weight_namespace,output,window,rope_base=1000000.,epsilon=1e-6):
    if any(type(v) is not int or v<1 for v in (tokens,width,heads,kv_heads,depth,inner,window)):
        raise ValueError('invalid transformer geometry')
    if width%heads or heads%kv_heads or (width//heads)%2:
        raise ValueError('invalid grouped-query geometry')
    t=tokens;d=width//heads;kv=kv_heads*d;ops=[];shapes={}
    def put(name,kind,kernel,inputs,outputs,params,weights=None):
        op={'id':name,'op':kind,'kernel':kernel,'returns_status':True,
            'graph_slots':{'inputs':inputs,'outputs':outputs},'params':params}
        if weights:op['weight_refs']=weights
        ops.append(op)
    def project(name,source,k,n,rows,prefix,bias=True):
        op=linear(name,source,name,k,n,rows,prefix)
        if not bias:
            op['op']='linear_rows_checked_no_bias';op['kernel']='linear_rows_checked_no_bias_f32'
            del op['weight_refs']['bias'];del op['params']['call_constants']['linear_bias_elements']
        ops.append(op);shapes[name]=[rows,n]
    def norm(name,source,prefix):
        shapes[name]=[t,width]
        put(name,'rmsnorm_rows_checked','rmsnorm_rows_checked_f32',{'input':source},{'output':name},
            {'R':t,'C':width,'eps':epsilon,'call_constants':{'input_elements':t*width,'input_stride':width,
            'gamma_elements':width,'output_elements':t*width,'output_stride':width,'rows':t,'width':width}},
            {'gamma':prefix+'.weight'})
    def add(name,left,right):
        shapes[name]=[t,width]
        put(name,'audio_scaled_sum_strided_checked','audio_scaled_sum_strided_f32_checked',
            {'left':left,'right':right},{'output':name},{'R':t,'C':width,'scale':1.,'call_constants':{
            'sum_left_elements':t*width,'sum_right_elements':t*width,'sum_output_elements':t*width,
            'sum_left_stride':width,'sum_right_stride':width,'sum_output_stride':width,'sum_rows':t,'sum_columns':width}})
    def layout(name,source,h,inverse=False):
        shapes[name]=[t,h*d] if inverse else [h,t,d]
        put(name,'token_head_layout_checked','layout_head_to_token_checked_f32' if inverse else 'layout_token_to_head_checked_f32',
            {'input':source},{'output':name},{'T':t,'H':h,'D':d,'call_constants':{
            'input_elements':t*h*d,'output_elements':t*h*d,'tokens':t,'heads':h,'head_dim':d}})
    previous=source
    for layer in range(depth):
        p=f'layer{layer}_';w=f'{weight_namespace}.layers.{layer}'
        norm(p+'norm1',previous,w+'.input_layernorm')
        for key,h,n in [('q',heads,width),('k',kv_heads,kv),('v',kv_heads,kv)]:
            project(p+key,p+'norm1',width,n,t,w+'.self_attn.'+key+'_proj')
            layout(p+key+'_head',p+key,h)
        put(p+'rope','rope_qk_bounded','rope_qk_split_bounded_f32',
            {'q':p+'q_head','k':p+'k_head'},{'q':p+'q_head','k':p+'k_head'},
            {'H':heads,'KV':kv_heads,'T':t,'D':d,'freq_base':rope_base,'call_constants':{
            'q_elements':t*width,'k_elements':t*kv,'heads':heads,'kv_heads':kv_heads,'tokens':t,'head_dim':d,'pos_offset':0}})
        for name in ('work','attention_head'):shapes[p+name]=[heads,t,d]
        put(p+'attention','attention_sliding_bounded','attention_sliding_head_major_bounded_f32',
            {'q':p+'q_head','k':p+'k_head','v':p+'v_head'},
            {'work':p+'work','out':p+'attention_head'},
            {'H':heads,'KV':kv_heads,'T':t,'D':d,'W':window,'call_constants':{
            'q_elements':t*width,'k_elements':t*kv,'v_elements':t*kv,'work_elements':t*width,
            'output_elements':t*width,'heads':heads,'kv_heads':kv_heads,'tokens':t,'head_dim':d,'sliding_window':window}})
        layout(p+'attention_token',p+'attention_head',heads,True)
        project(p+'attention_out',p+'attention_token',width,width,t,w+'.self_attn.o_proj',False)
        add(p+'residual',previous,p+'attention_out');norm(p+'norm2',p+'residual',w+'.post_attention_layernorm')
        for key in ('gate','up'):project(p+key,p+'norm2',width,inner,t,w+'.mlp.'+key+'_proj',False)
        shapes[p+'activated']=[t,inner]
        put(p+'activated','silu_mul_rows_checked','silu_mul_rows_checked_f32',{'gate':p+'gate','up':p+'up'},
            {'output':p+'activated'},{'N':t*inner,'call_constants':{'gate_elements':t*inner,'up_elements':t*inner,
            'output_elements':t*inner,'elements':t*inner}})
        project(p+'down',p+'activated',inner,width,t,w+'.mlp.down_proj',False)
        add(p+'output',p+'residual',p+'down');previous=p+'output'
    norm(output,previous,weight_namespace+'.norm')
    return ops,shapes
