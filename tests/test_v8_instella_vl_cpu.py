"""Geometry and prefix ordering for the bounded Instella CPU host."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "version/v8/scripts"))
from run_instella_vl_cpu_v8 import GRID, MEAN, STD, WIDTH, bridge, image_tiles


class InstellaHostTests(unittest.TestCase):
    def test_overview_and_centered_tile_are_distinct(self):
        image = Image.new("RGB", (224, 112), (255, 0, 0))
        pixels, grid = image_tiles(image, [(336, 336), (672, 336)])
        self.assertEqual(grid, (1, 1))
        self.assertEqual(pixels.shape, (2, 3, 336, 336))
        red = (np.array([1, 0, 0], dtype=np.float32) - MEAN) / STD
        black = -MEAN / STD
        np.testing.assert_array_equal(pixels[0, :, 0, 0], red)
        np.testing.assert_array_equal(pixels[1, :, 0, 0], black)
        np.testing.assert_array_equal(pixels[1, :, 168, 0], red)

    def test_bridge_reorders_tiles_and_inserts_row_separator(self):
        projected = np.empty((3, GRID * GRID, WIDTH), np.float32)
        projected[0] = -1
        for tile in range(2):
            for row in range(GRID):
                projected[tile + 1, row * GRID:(row + 1) * GRID] = tile * 100 + row
        newline = np.full(WIDTH, -2, dtype=np.float32)
        actual = bridge(projected, newline, (672, 336), (2, 1))
        self.assertEqual(actual.shape, (GRID * GRID + GRID * (2 * GRID + 1), WIDTH))
        self.assertTrue((actual[:GRID * GRID] == -1).all())
        rows = actual[GRID * GRID:].reshape(GRID, 2 * GRID + 1, WIDTH)
        for row in range(GRID):
            self.assertTrue((rows[row, :GRID] == row).all())
            self.assertTrue((rows[row, GRID:2 * GRID] == 100 + row).all())
            np.testing.assert_array_equal(rows[row, -1], newline)

    def test_bridge_unpads_and_rejects_missing_tile(self):
        projected = np.zeros((2, GRID * GRID, WIDTH), np.float32)
        projected[1] = np.repeat(np.arange(GRID), GRID)[:, None]
        newline = np.ones(WIDTH, np.float32)
        out = bridge(projected, newline, (224, 112), (1, 1))
        self.assertEqual(out.shape, (876, WIDTH))
        self.assertEqual(out[576, 0], 6)
        self.assertEqual(out[-2, 0], 17)
        with self.assertRaisesRegex(ValueError, "tile/newline shape"):
            bridge(projected[:1], newline, (224, 112), (1, 1))


if __name__ == "__main__":
    unittest.main()
