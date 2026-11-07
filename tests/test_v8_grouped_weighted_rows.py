"""Model-neutral grouped pooling: numerical order, strides and atomic rejection."""
import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)


def ordered_reference(options, fields, counts):
    result = np.empty_like(fields)
    first = 0
    for q, count in enumerate(counts):
        logits = []
        for row in options[first:first + count]:
            dot = 0.0
            for a, b in zip(row, fields[q]):
                dot += float(a) * float(b)
            logits.append(np.float32(dot * (1.0 / math.sqrt(options.shape[1]))))
        maximum = max(float(x) for x in logits)
        exps = [np.float32(math.exp(float(x) - maximum)) for x in logits]
        denominator = 0.0
        for x in exps:
            denominator += float(x)
        probs = [np.float32(float(x) / denominator) for x in exps]
        for c in range(options.shape[1]):
            total = 0.0
            for i, p in enumerate(probs):
                total += float(p) * float(options[first + i, c])
            result[q, c] = np.float32(total)
        first += count
    return result


class GroupedWeightedRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.libs = []
        for optimization in ('-O0', '-O3'):
            path = Path(cls.temp.name) / (optimization + '.so')
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                            '-shared', '-fPIC', '-ffp-contract=fast',
                            '-I', str(ROOT / 'include'),
                            str(ROOT / 'src/kernels/grouped_weighted_rows.c'),
                            '-lm', '-o', str(path)], check=True)
            lib = ctypes.CDLL(str(path))
            fn = lib.ck_grouped_softmax_weighted_rows_f32
            s = ctypes.c_size_t
            fn.argtypes = [FP,s,s,FP,s,s,IP,s,FP,s,s,FP,s,s,s,s]
            fn.restype = ctypes.c_int
            cls.libs.append((lib, fn))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        rng = np.random.default_rng(814)
        self.options = rng.normal(size=(5, 7)).astype('f4')
        self.fields = rng.normal(size=(2, 6)).astype('f4')
        self.counts = np.array([3, 2], dtype='i4')
        self.output = np.full((2, 8), -91, dtype='f4')
        self.scratch = np.zeros(15, dtype='f4')
        self.options[:, 5:] = np.nan
        self.fields[:, 5:] = np.nan

    def args(self):
        return [self.options.ctypes.data_as(FP), 35, 7,
                self.fields.ctypes.data_as(FP), 12, 6,
                self.counts.ctypes.data_as(IP), 2,
                self.output.ctypes.data_as(FP), 16, 8,
                self.scratch.ctypes.data_as(FP), 15, 2, 5, 5]

    def test_ordered_fp64_oracle_strides_and_optimization_levels(self):
        expected = ordered_reference(self.options[:, :5], self.fields[:, :5], self.counts)
        for _, fn in self.libs:
            self.assertEqual(fn(*self.args()), 0)
            np.testing.assert_array_equal(self.output[:, :5], expected)
            np.testing.assert_array_equal(self.output[:, 5:], -91)

    def test_rejection_before_parity_and_reuse(self):
        fn = self.libs[1][1]
        for index, value in [(0,None),(3,None),(6,None),(8,None),(11,None),
                             (1,32),(4,10),(7,1),(9,12),(12,14),
                             (2,4),(5,4),(10,4),(13,0),(14,0),(15,0),
                             (13,2**63),(15,2**63)]:
            with self.subTest(index=index, value=value):
                args = self.args(); args[index] = value
                self.assertNotEqual(fn(*args), 0)
                np.testing.assert_array_equal(self.output, -91)
        for counts in ([0,5],[-1,6],[2,2],[4,2]):
            self.counts[:] = counts
            self.assertNotEqual(fn(*self.args()), 0)
            np.testing.assert_array_equal(self.output, -91)
        self.counts[:] = [3,2]
        # Exact minimal row extents are accepted; one-short rejected above.
        args = self.args(); args[1]=33; args[4]=11; args[9]=13
        self.assertEqual(fn(*args), 0)
        np.testing.assert_array_equal(self.output[:, :5], ordered_reference(
            self.options[:, :5], self.fields[:, :5], self.counts))

    def test_partial_overlap_misalignment_and_finite_overflow(self):
        fn = self.libs[1][1]
        for index, pointer in [(8,ctypes.cast(self.options.ctypes.data+4,FP)),
                               (11,ctypes.cast(self.output.ctypes.data+4,FP)),
                               (0,ctypes.cast(self.options.ctypes.data+1,FP)),
                               (6,ctypes.cast(self.counts.ctypes.data+1,IP))]:
            args=self.args(); args[index]=pointer
            before=self.options.copy()
            self.assertNotEqual(fn(*args),0)
            np.testing.assert_array_equal(self.options,before)
            np.testing.assert_array_equal(self.output,-91)
        saved=self.options.copy()
        for bad in (np.nan,np.inf):
            self.options[0,0]=bad
            self.assertNotEqual(fn(*self.args()),0)
            np.testing.assert_array_equal(self.output,-91)
        self.options[:]=saved
        self.options[3:,:5]=np.finfo('f4').max
        saved_fields=self.fields.copy(); self.fields[1,:5]=np.finfo('f4').max
        self.assertNotEqual(fn(*self.args()),0)
        np.testing.assert_array_equal(self.output,-91)
        self.options[:]=saved; self.fields[:]=saved_fields
        self.assertEqual(fn(*self.args()),0)
        np.testing.assert_array_equal(self.output[:,:5],ordered_reference(
            self.options[:,:5],self.fields[:,:5],self.counts))

    def test_independent_torch_and_within_group_reordering(self):
        try:
            import torch
        except ImportError:
            self.skipTest('independent PyTorch comparison unavailable')
        fn=self.libs[1][1]
        options=torch.from_numpy(self.options[:,:5].copy())
        fields=torch.from_numpy(self.fields[:,:5].copy())
        expected=[]; first=0
        for q,n in enumerate(self.counts):
            rows=options[first:first+n]
            expected.append((torch.softmax(rows @ fields[q] / math.sqrt(5),dim=0)[:,None]*rows).sum(0))
            first+=n
        self.assertEqual(fn(*self.args()),0)
        baseline=self.output[:,:5].copy()
        np.testing.assert_allclose(baseline,torch.stack(expected).numpy(),atol=2e-7,rtol=2e-7)
        self.options[:]=self.options[[2,0,1,4,3]].copy()
        self.assertEqual(fn(*self.args()),0)
        np.testing.assert_allclose(self.output[:,:5],baseline,atol=2e-7,rtol=0)

    def test_singletons_and_multiple_group_geometries(self):
        fn = self.libs[1][1]
        rng = np.random.default_rng(808)
        for counts, width in (([1],1),([1,4,2],9),([2,1,1,3],17)):
            with self.subTest(counts=counts,width=width):
                groups, rows = len(counts), sum(counts)
                options = rng.normal(size=(rows,width)).astype('f4')
                fields = rng.normal(size=(groups,width)).astype('f4')
                group_counts = np.array(counts,dtype='i4')
                output = np.zeros((groups,width),dtype='f4')
                scratch = np.zeros(rows+groups*width,dtype='f4')
                self.assertEqual(fn(options.ctypes.data_as(FP),options.size,width,
                    fields.ctypes.data_as(FP),fields.size,width,
                    group_counts.ctypes.data_as(IP),groups,
                    output.ctypes.data_as(FP),output.size,width,
                    scratch.ctypes.data_as(FP),scratch.size,groups,rows,width),0)
                np.testing.assert_array_equal(output,ordered_reference(options,fields,counts))

    def test_model_neutral_generated_call_and_rejection(self):
        from tests.test_v8_token_span_mean_checked import _circuit
        from tests.v8_checked_graph_test_support import compile_native_graph
        graph=_circuit()
        graph['name']='grouped_weighted_rows_synthetic'
        graph['native_entry']['function']='ck_test_grouped_rows'
        shapes={'options':[5,5],'fields':[2,5],'group_counts':[2],'output':[2,5]}
        graph['activation_buffers']={k:{'shape':v} for k,v in shapes.items()}
        graph['activation_bindings']={k:k for k in shapes}
        graph['block_types']['pool']['body']['ops']=[{
            'id':'pool','op':'grouped_softmax_weighted_rows_checked',
            'kernel':'grouped_softmax_weighted_rows_checked_f32','returns_status':True,
            'graph_slots':{'inputs':{k:'external:'+k for k in ('options','fields','group_counts')},
                           'outputs':{'output':'external:output'}},
            'params':{'G':2,'O':5,'C':5,'X':15,'call_constants':{
                'option_elements':25,'option_stride':5,'field_elements':10,'field_stride':5,
                'count_elements':2,'output_elements':10,'output_stride':5,
                'scratch_elements':15,'groups':2,'option_rows':5,'channels':5}}}]
        source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,
            'embed_dim':5,'num_heads':1,'num_kv_heads':1,'head_dim':5,'intermediate_size':5,
            'context_length':5,'max_seq_len':5,'vocab_size':16,
            'activation_buffer_dtypes':{'group_counts':'i32'}},
            'entries':[],'quant_summary':{},'template':graph}
        with tempfile.TemporaryDirectory() as temp:
            layout,calls,_,_,execute=compile_native_graph(Path(temp),source,graph)
            self.assertEqual(calls['errors'],[])
            self.assertEqual([op['function'] for op in calls['operations']],
                             ['ck_grouped_softmax_weighted_rows_f32'])
            size=layout['memory']['arena']['total_size']
            raw=(ctypes.c_uint8*(size+63))()
            arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
            entries={x['name']:x for x in layout['memory']['activations']['buffers']}
            arrays={k:np.ndarray(v,dtype='i4' if k=='group_counts' else 'f4',buffer=arena,
                                offset=entries[k]['abs_offset']) for k,v in shapes.items()}
            arrays['options'][:]=self.options[:,:5]
            arrays['fields'][:]=self.fields[:,:5]
            arrays['group_counts'][:]=[3,1]
            arrays['output'][:]=-91
            self.assertNotEqual(execute(arena,size),0)
            np.testing.assert_array_equal(arrays['output'],-91)
            arrays['group_counts'][:]=self.counts
            self.assertEqual(execute(arena,size),0)
            np.testing.assert_array_equal(arrays['output'],ordered_reference(
                self.options[:,:5],self.fields[:,:5],self.counts))


if __name__ == '__main__':
    unittest.main()
