"""Reject malformed dense weight artifacts before constructing training IR."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'version/v7/scripts'))
from build_ir_train_v7 import _manifest_weight_index

@pytest.mark.parametrize('dtype,width', [('fp32',4), ('bf16',2), ('fp16',2)])
def test_manifest_dense_extents(dtype, width):
    entry = dict(name='projection', dtype=dtype, shape=[6,3], size=18*width)
    assert _manifest_weight_index({'entries':[entry]})['projection'] == entry
    # Reproduces concatenated gate/up byte size with only the gate shape.
    entry['shape'] = [3,3]
    with pytest.raises(ValueError, match='shape and byte size disagree'):
        _manifest_weight_index({'entries':[entry]})


def test_legacy_absent_shapes_and_packed_formats_are_not_dense_extents():
    entries = [dict(name='legacy', dtype='fp32', size=24),
               dict(name='packed', dtype='q4_k', shape=[256,256], size=36864)]
    assert len(_manifest_weight_index({'entries':entries})) == 2
