"""Independent reduction and rejection checks for contiguous gather/scatter."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class IndexedRowsTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        target = Path(cls.tmp.name) / "rows.so"
        subprocess.run(
            [
                "cc",
                "-O2",
                "-shared",
                "-fPIC",
                "-I" + str(ROOT / "include"),
                str(ROOT / "src/kernels/indexed_rows_train.c"),
                "-lm",
                "-o",
                str(target),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(target))
        cls.fp = ctypes.POINTER(ctypes.c_float)
        cls.ip = ctypes.POINTER(ctypes.c_int32)
        z = ctypes.c_size_t
        cls.lib.ck_indexed_rows_f32.argtypes = [
            cls.fp,
            z,
            cls.ip,
            z,
            cls.fp,
            z,
            z,
            z,
            z,
        ]
        cls.lib.ck_indexed_rows_backward_f32.argtypes = [
            cls.fp,
            z,
            cls.ip,
            z,
            cls.fp,
            z,
            cls.fp,
            z,
            z,
            z,
            z,
        ]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backward(self, dy, ids, out, scratch, **kw):
        return self.lib.ck_indexed_rows_backward_f32(
            dy.ctypes.data_as(self.fp),
            kw.get("dy_count", dy.size),
            ids.ctypes.data_as(self.ip),
            kw.get("id_count", ids.size),
            out.ctypes.data_as(self.fp),
            kw.get("out_count", out.size),
            scratch.ctypes.data_as(self.fp),
            kw.get("scratch_count", scratch.size),
            len(ids),
            4,
            2,
        )

    def test_fp64_order_and_repeated_indices(self):
        # FP32 left-to-right would lose the middle contribution.
        dy = np.array([[1e8, 3], [1, 4], [-1e8, 5]], dtype="f4")
        ids = np.array([2, 2, 2], dtype="i4")
        out = np.full((5, 2), -99, dtype="f4")
        scratch = np.zeros((4, 2), dtype="f4")
        self.assertEqual(self.backward(dy, ids, out, scratch), 0)
        expected = np.zeros((4, 2), dtype="f4")
        for c in range(2):
            acc = 0.0
            for r in range(3):
                acc += float(dy[r, c])
            expected[2, c] = acc
        np.testing.assert_array_equal(out[:4], expected)
        np.testing.assert_array_equal(out[4], -99)

    def test_reject_then_reuse(self):
        dy = np.ones((3, 2), dtype="f4")
        ids = np.array([0, 2, 2], dtype="i4")
        out = np.full((4, 2), -99, dtype="f4")
        scratch = np.zeros_like(out)
        for kw in (
            {"dy_count": 5},
            {"id_count": 2},
            {"out_count": 7},
            {"scratch_count": 7},
        ):
            self.assertNotEqual(self.backward(dy, ids, out, scratch, **kw), 0)
            np.testing.assert_array_equal(out, -99)
        ids[2] = 4
        self.assertNotEqual(self.backward(dy, ids, out, scratch), 0)
        np.testing.assert_array_equal(out, -99)
        ids[2] = 2
        dy[0, 0] = np.nan
        self.assertNotEqual(self.backward(dy, ids, out, scratch), 0)
        np.testing.assert_array_equal(out, -99)
        dy[0, 0] = 1
        self.assertNotEqual(self.backward(dy, ids, out, out), 0)
        np.testing.assert_array_equal(out, -99)
        self.assertEqual(self.backward(dy, ids, out, scratch), 0)
        np.testing.assert_array_equal(out, [[1, 1], [0, 0], [2, 2], [0, 0]])

    def test_overflow_is_atomic(self):
        dy = np.full((3, 2), np.finfo("f4").max, dtype="f4")
        ids = np.zeros(3, dtype="i4")
        out = np.full((4, 2), -99, dtype="f4")
        scratch = np.zeros_like(out)
        self.assertNotEqual(self.backward(dy, ids, out, scratch), 0)
        np.testing.assert_array_equal(out, -99)

    def test_gather_and_invalid_ids(self):
        table = np.arange(8, dtype="f4").reshape(4, 2)
        ids = np.array([3, 1, 3], dtype="i4")
        out = np.full((4, 2), -99, dtype="f4")

        def call():
            return self.lib.ck_indexed_rows_f32(
                table.ctypes.data_as(self.fp),
                table.size,
                ids.ctypes.data_as(self.ip),
                len(ids),
                out.ctypes.data_as(self.fp),
                out.size,
                3,
                4,
                2,
            )

        self.assertEqual(call(), 0)
        np.testing.assert_array_equal(out[:3], table[ids])
        np.testing.assert_array_equal(out[3], -99)
        baseline = out.copy()
        ids[1] = -1
        self.assertNotEqual(call(), 0)
        np.testing.assert_array_equal(out, baseline)


if __name__ == "__main__":
    unittest.main()
