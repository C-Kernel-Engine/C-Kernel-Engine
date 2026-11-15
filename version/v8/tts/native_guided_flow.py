"""Typed binding for the native guided-flow driver; no Python sampling loop."""
import ctypes as C,json
import numpy as np
P=C.POINTER(C.c_float);U=C.POINTER(C.c_uint8);S=C.c_size_t
ENTRY=C.CFUNCTYPE(C.c_int,U,S)
NOISE=C.CFUNCTYPE(C.c_int,P,S,S,C.c_uint64,C.c_uint64)
CANCEL=C.CFUNCTYPE(C.c_int,C.c_void_p)
class Model(C.Structure):
    _fields_=[('entry',ENTRY),('arena',U),('arena_bytes',S)]+[(k,S) for k in ('latent','time','condition','history','prediction')]
class Update(C.Structure):
    _fields_=[('entry',ENTRY),('arena',U),('arena_bytes',S)]+[(k,S) for k in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','amplitude','output')]
class Request(C.Structure):
    _fields_=[('model',Model),('update',Update),('noise',NOISE)]+[(k,P) for k in ('condition','history','times')]+[(k,S) for k in ('latent_elements','condition_elements','history_elements','steps')]+[(k,C.c_float) for k in ('cfg','sigma','temperature')]+[(k,C.c_uint64) for k in ('seed','stream')]+[('workspace',P),('workspace_elements',S),('output',P),('output_elements',S),('should_cancel',CANCEL),('cancel_context',C.c_void_p)]

def offsets(folder):
    layout=json.loads((folder/'layout.json').read_text())
    return {b['name']:b['abs_offset'] for b in layout['memory']['activations']['buffers']}

def schedule():
    """Pinned 10-step EPSS+sway FP32 control coefficients, not model activations."""
    t=np.array([0,2,4,6,8,12,16,20,24,28,32],dtype='f4')/np.float32(32)
    return (t-(np.cos(np.float32(np.pi/2)*t)-np.float32(1)+t)).astype('f4')

class NativeGuidedFlow:
    """Own live native libraries and serialize access to their reusable arenas."""
    def __init__(self, model_bundle, graph, model_folder, update_folder):
        import threading
        from version.v8.tts.replay_ming3_dit_block import compile_block
        from version.v8.tts.build_ming3_sampler_update_circuit import build_circuit
        inv=graph['contract']['runtime_invariants']
        if not inv.get('complete_dit') or graph['activation_buffers']['output']['shape']!=[3,inv['latent']]:
            raise ValueError('native sampler requires the complete three-row DiT contract')
        self.model_bundle=model_bundle;self.graph=graph;self.lock=threading.Lock()
        self.latent=graph['contract']['runtime_invariants']['latent']
        self.condition=graph['contract']['runtime_invariants']['condition']
        self.update_bundle=compile_block(update_folder,build_circuit(self.latent),{},
            runtime_sources=['src/runtime/checked_guided_flow.c'],
            extra_exports=['ck_run_checked_guided_flow','audio_noise_counter_gaussian_f32'])
        self.model_offsets=offsets(model_folder);self.update_offsets=offsets(update_folder)
        self.driver=self.update_bundle[3].ck_run_checked_guided_flow
        self.driver.argtypes=[C.POINTER(Request)];self.driver.restype=C.c_int
    def run(self,condition,history,*,seed=731,stream=0,cfg=2.,sigma=.25,temperature=0.,should_cancel=None):
        for value in (seed,stream):
            if type(value) is not int or not 0<=value<2**64:raise ValueError('seed and stream must be uint64 integers')
        cond=np.ascontiguousarray(condition,dtype='f4').reshape(-1).copy()
        hist=np.ascontiguousarray(history,dtype='f4').reshape(-1).copy()
        if cond.size!=self.condition or hist.size!=self.latent:raise ValueError('conditioning/history shape mismatch')
        times=schedule();workspace=np.empty(11*self.latent,'f4');output=np.full(self.latent,np.nan,'f4')
        m=self.model_offsets;u=self.update_offsets;mb=self.model_bundle;ub=self.update_bundle
        model=Model(C.cast(mb[0],ENTRY),C.cast(mb[1],U),len(mb[1]),
            m['latent'],m['time'],m['condition'],m['history'],m['output']+2*self.latent*4)
        update=Update(ENTRY(('ck_ming3_sampler_update',ub[3])),C.cast(ub[1],U),len(ub[1]),
            *[u[k] for k in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output')])
        callback_errors=[]
        def check_cancel(context):
            try:return int(bool(should_cancel()))
            except BaseException as exc:
                callback_errors.append(exc);return 1
        cancelled=CANCEL(check_cancel) if should_cancel else CANCEL()
        request=Request(model,update,NOISE(('audio_noise_counter_gaussian_f32',ub[3])),
            cond.ctypes.data_as(P),hist.ctypes.data_as(P),times.ctypes.data_as(P),
            self.latent,self.condition,self.latent,10,cfg,sigma,temperature,seed,stream,
            workspace.ctypes.data_as(P),workspace.size,output.ctypes.data_as(P),output.size,cancelled,None)
        with self.lock:status=self.driver(C.byref(request))
        if callback_errors:raise callback_errors[0]
        if status:raise RuntimeError(f'native guided flow returned {status}')
        return output
