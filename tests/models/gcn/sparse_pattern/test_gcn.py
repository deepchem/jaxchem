import unittest

import haiku as hk
import jax.numpy as jnp
import jax.random as jrandom
from jaxchem.models import SparseGCN


# params
in_feats = 64
hidden_feats = [64, 32, 16]
node_num = 10


class TestSparseGCN(unittest.TestCase):
    """Test SparseGCN"""

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
        model = SparseGCN(in_feats=in_feats, hidden_feats=hidden_feats)
        return model(node_feats, adj, is_training)

    def test_forward_shape(self):
        """Test output shape of SparseGCN"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (node_num, hidden_feats[-1])
