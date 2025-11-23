import unittest

import numpy as np

from framed import discretize_frame


class TestDiscretizeFrame(unittest.TestCase):
    def test_discretize_frame_default_mesh(self):
        nodes, elements = discretize_frame()

        expected_nodes = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [0.0, 4.0],
                [0.0, 5.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
                [0.2, 5.0],
                [0.4, 5.0],
                [0.6, 5.0],
                [0.8, 5.0],
            ]
        )

        expected_elements = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [5, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 11],
            ]
        )

        self.assertEqual(nodes.shape, (16, 2))
        self.assertEqual(elements.shape, (15, 2))
        np.testing.assert_allclose(nodes, expected_nodes)
        np.testing.assert_array_equal(elements, expected_elements)


if __name__ == "__main__":
    unittest.main()
