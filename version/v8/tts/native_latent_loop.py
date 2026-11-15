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
        ('output_elements',S),('valid_frames',C.POINTER(S))]
