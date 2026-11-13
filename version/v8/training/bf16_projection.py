"""Opt-in fixed-capacity projection precision authoring for the shared trainer.

FP32 masters and activation gradients; RNE BF16 saved compute operands; FP32
output and ordered accumulation. This is not a global BF16/autocast switch.
"""
from copy import deepcopy


def use_bf16_projection(ir, op_id):
    """Return IR with one biasless projection using the declared mixed VJP."""
    result = deepcopy(ir)
    matches = [o for o in result['ops'] if o['op_id'] == op_id]
    if len(matches) != 1:
        raise ValueError('expected one projection operation')
    op = matches[0]
    if op.get('kernel_id') != 'gemm_blocked_serial' or set(op['weights']) != {'W'}:
        raise ValueError('only explicit biasless FP32 GEMM projections are supported')
    x = op['dataflow']['inputs']['input']
    w = op['weights']['W']
    outputs = op['dataflow']['outputs']
    if set(outputs) != {'y'}:
        raise ValueError('expected one projection output')
    y = outputs['y']
    if any(ref.get('dtype') != 'fp32' for ref in (x, w, y)):
        raise ValueError('projection masters and external activations must be FP32')
    if any(len(ref.get('shape', [])) != 2 for ref in (x, w, y)):
        raise ValueError('expected bounded row-major matrices')
    rows, inputs = x['shape']; out, wi = w['shape']
    if any(type(n) is not int or n <= 0 for n in (rows, inputs, out)) or wi != inputs or y['shape'] != [rows, out]:
        raise ValueError('inconsistent projection geometry')
    def aux(name, shape, dtype):
        tid = 'aux.projection_%s.%s' % (op_id, name)
        if tid in result['tensors']:
            raise ValueError('projection auxiliary identity collision')
        n = 1
        for d in shape: n *= d
        meta = dict(shape=shape, numel=n, dtype=dtype, kind='aux', requires_grad=False, persistent=False)
        result['tensors'][tid] = meta
        return dict(meta, tensor=tid)
    sx = aux('saved_input', [rows, inputs], 'bf16')
    sw = aux('saved_weight', [out, inputs], 'bf16')
    scratch = aux('scratch', [rows*out], 'fp32')
    op.update(kernel_id='projection_bf16_train_forward', map_call=True,
              grad_rule='projection_bf16_train_backward',
              weights={'weight': w},
              dataflow=dict(inputs={'input':x}, outputs=dict(output=y, saved_input=sx, saved_weight=sw, scratch=scratch)),
              save_for_backward=dict(saved_input=sx, saved_weight=sw),
              params=dict(call_constants=dict(rows=rows, inputs=inputs, outputs=out,
                  input_elements=rows*inputs, weight_elements=out*inputs, output_elements=rows*out,
                  saved_input_elements=rows*inputs, saved_weight_elements=out*inputs,
                  d_output_elements=rows*out, d_input_elements=rows*inputs, d_weight_elements=out*inputs,
                  scratch_elements=rows*out, backward_scratch_elements=(rows+out)*inputs)))
    result['config']['train_batch_policy'] = 'fixed'
    return result
