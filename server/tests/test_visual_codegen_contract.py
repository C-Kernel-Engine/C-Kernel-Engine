"""Contract emitted from circuit configuration, not model-name dispatch."""
from pathlib import Path
import importlib.util
import sys
import pytest


def test_visual_width_includes_deepstack_and_rejects_missing_slices():
    path=Path(__file__).resolve().parents[2]/'version/v8/scripts/codegen_v8.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('visual_codegen_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    emit=module._emit_visual_prefix_contract
    assert emit({'embed_dim':256})==''
    cfg={'embed_dim':256,'num_deepstack_layers':3,
         'multimodal_bridge_contract':{'prefix_policy':'declared'}}
    assert 'return 1024;' in emit(cfg)
    assert 'return 1280;' in emit(dict(cfg,input_embed_dim=1280))
    with pytest.raises(ValueError):emit(dict(cfg,input_embed_dim=256))
    with pytest.raises(ValueError):emit(dict(cfg,num_deepstack_layers=-1))
