"""Independent integer arithmetic oracle for bounded runtime extent production."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeExtentKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        library = Path(cls.temp_dir.name) / "libruntime_extent.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(ROOT / "src" / "kernels" / "runtime_extent.c"),
            "-o", str(library),
        ], check=True)
        native = ctypes.CDLL(str(library))
        cls.function = native.ck_runtime_sum_i32_checked
        cls.function.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        cls.function.restype = ctypes.c_int
        cls.copy_valid = native.ck_runtime_copy_valid_f32
        cls.copy_valid.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_size_t,
        ]
        cls.copy_valid.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def invoke(self, items, *, count=None, min_value=0, max_value=100,
               initial=0, capacity=100, value_elements=None):
        values = (ctypes.c_int32 * len(items))(*items)
        output = ctypes.c_int32(0xC0FFEE)
        status = self.function(values, len(items) if value_elements is None else value_elements,
                               len(items) if count is None else count,
                               min_value, max_value, initial, capacity,
                               ctypes.byref(output))
        return status, output.value

    def test_python_integer_oracle_and_repeated_lengths(self):
        for items, initial, capacity in [
            ([], 0, 0), ([1], 0, 1), ([1, 3, 2, 1, 4], 0, 11),
            ([3, 4, 5], 7, 19), ([0, 0, 0], 0, 0),
        ]:
            with self.subTest(items=items):
                status, valid = self.invoke(items, initial=initial, capacity=capacity)
                self.assertEqual(status, 0)
                self.assertEqual(valid, initial + sum(items))
        values = [2, 4, 1]
        for count, expected in [(1, 2), (3, 7), (2, 6), (0, 0)]:
            status, valid = self.invoke(values, count=count, capacity=7)
            self.assertEqual((status, valid), (0, expected))

    def test_rejects_before_output_write(self):
        size_max = ctypes.c_size_t(-1).value
        for name, items, kwargs, status in [
            ("excess_capacity", [2, 2], {"capacity": 3}, -2),
            ("count_exceeds_input", [1], {"count": 2}, -2),
            ("invalid_negative", [-1], {}, -1),
            ("below_min", [0], {"min_value": 1}, -1),
            ("above_max", [101], {}, -1),
            ("invalid_range", [1], {"min_value": 2, "max_value": 1}, -1),
            ("initial_over_capacity", [], {"initial": 2, "capacity": 1}, -2),
            ("sum_overflow", [1], {"initial": size_max, "capacity": size_max}, -3),
            ("metadata_overflow", [1], {"count": size_max,
                                      "value_elements": size_max}, -3),
        ]:
            with self.subTest(name=name):
                actual, untouched = self.invoke(items, **kwargs)
                self.assertEqual(actual, status)
                self.assertEqual(untouched, 0xC0FFEE)

    def test_copy_valid_columns_and_preserve_padding(self):
        for frames in (0, 1, 5, 8):
            with self.subTest(frames=frames):
                channels, input_stride, output_stride = 3, 11, 10
                source = (ctypes.c_float * (channels * input_stride))(*[
                    float(channel * 100 + column)
                    for channel in range(channels) for column in range(input_stride)
                ])
                target = (ctypes.c_float * (channels * output_stride))(*(
                    [-999.0] * (channels * output_stride)))
                self.assertEqual(self.copy_valid(
                    source, len(source), channels, frames, input_stride,
                    target, len(target), output_stride), 0)
                expected = [-999.0] * len(target)
                for channel in range(channels):
                    for frame in range(frames):
                        expected[channel * output_stride + frame] = (
                            channel * 100 + frame)
                self.assertEqual(list(target), expected)

    def test_copy_rejects_bad_capacity_and_stride_before_write(self):
        source = (ctypes.c_float * 22)(*range(22))
        target = (ctypes.c_float * 22)(*([-999.0] * 22))
        cases = [
            (18, 2, 8, 11, len(target), 11, -2),
            (len(source), 2, 8, 11, 18, 11, -2),
            (len(source), 2, 8, 7, len(target), 11, -1),
            (len(source), 2, 8, 11, len(target), 7, -1),
            (len(source), 2, 8, ctypes.c_size_t(-1).value,
             len(target), 11, -3),
        ]
        for input_count, channels, frames, input_stride, output_count, output_stride, expected in cases:
            with self.subTest(input_stride=input_stride, output_count=output_count):
                self.assertEqual(self.copy_valid(
                    source, input_count, channels, frames, input_stride,
                    target, output_count, output_stride), expected)
                self.assertEqual(list(target), [-999.0] * len(target))


if __name__ == "__main__":
    unittest.main()
