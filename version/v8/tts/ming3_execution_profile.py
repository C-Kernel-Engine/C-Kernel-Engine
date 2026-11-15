"""Load the versioned model execution policy; no runtime arithmetic or offsets."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def load_profile(*,cached=True):
    name='ming3_synthesis_fp32_v1.json' if cached else 'ming3_synthesis_fp32_full_prefix_v1.json'
    return json.loads((ROOT/'version/v8/contracts'/name).read_text())
def sampling_times():
    import numpy as np
    return np.asarray(load_profile()['sampling']['times'],dtype='f4')
