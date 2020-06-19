import unittest

import pytest
import numpy as np
from jaxchem.models.nn import pad_graph_pooling


class TestGraphPooling(unittest.TestCase):
    """Test graph pooling function."""

    def test_pad_graph_pooling(self):
        "Test pad_graph_pooling function."
        pad_graph_data = np.array([
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [2.0, 2.0, 2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])
        input_shape = pad_graph_data.shape

        # mean pool
        mean_pool = pad_graph_pooling('mean')
        mean_out = mean_pool(pad_graph_data)
        assert mean_out.shape == (input_shape[0], input_shape[2])
        maen_out_true = np.array([
            [0.25, 0.25, 0.25, 0.25, 0.25],
            [0.50, 0.50, 0.50, 0.50, 0.50],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        np.testing.assert_allclose(mean_out, maen_out_true)

        # sum pool
        sum_pool = pad_graph_pooling('sum')
        sum_out = sum_pool(pad_graph_data)
        assert sum_out.shape == (input_shape[0], input_shape[2])
        sum_out_true = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ])
        np.testing.assert_allclose(sum_out, sum_out_true)

        # min pool
        min_pool = pad_graph_pooling('min')
        min_out = min_pool(pad_graph_data)
        assert min_out.shape == (input_shape[0], input_shape[2])
        min_out_true = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(min_out, min_out_true)

        # max pool
        max_pool = pad_graph_pooling('max')
        max_out = max_pool(pad_graph_data)
        assert max_out.shape == (input_shape[0], input_shape[2])
        max_out_true = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ])
        np.testing.assert_allclose(max_out, max_out_true)

    def test_invalid_args(self):
        "Test invalid argument."
        # pad_graph_pooling
        with pytest.raises(ValueError):
            invalid_pad_graph_pooling = pad_graph_pooling('set2set')
