import unittest

import haiku as hk
import jax.numpy as jnp
import jax.random as jrandom
from jaxchem.models import SparseGCNLayer


# params
in_feats = 64
out_feats = 32
node_num = 10


class TestSparseGCNLayer(unittest.TestCase):
    """Test SparseGCNLayer"""

    def setup_method(self, method):
        self.key = hk.PRNGSequence(1234)
        self.input_data = self.__setup_data()

    def __setup_data(self):
        node_feats = jrandom.normal(next(self.key), (node_num, in_feats))
        adj = jnp.array([
            [0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9],
            [4, 7, 3, 2, 1, 1, 8, 0, 8, 6, 5, 9, 0, 3, 5, 6],
        ])
        return (node_feats, adj, True)

    def __forward(self, node_feats, adj, is_training):
        model = SparseGCNLayer(in_feats=in_feats, out_feats=out_feats)
        return model(node_feats, adj, is_training)

    def __forward_with_batch_norm(self, node_feats, adj, is_training):
        model = SparseGCNLayer(in_feats=in_feats, out_feats=out_feats, batch_norm=True)
        return model(node_feats, adj, is_training)

    def test_forward_shape(self):
        """Test output shape of SparseGCNLayer"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (node_num, out_feats)

    def test_forward_shape_with_batch_norm(self):
        """Test output shape of SparseGCNLayer with BatchNorm"""
        forward = hk.transform_with_state(self.__forward_with_batch_norm)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (node_num, out_feats)
