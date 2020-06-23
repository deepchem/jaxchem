import unittest

import haiku as hk
import jax.numpy as jnp
import jax.random as jrandom
from jaxchem.models import SparseGCNPredicator


# params
in_feats = 64
hidden_feats = [64, 32, 16]
node_num = 10
pooling_method = 'mean'
predicator_hidden_feats = 16
n_out = 1
batch_size = 2


class TestSparseGCNPredicator(unittest.TestCase):
    """Test SparseGCNPredicator"""

    def setup_method(self, method):
        self.key = hk.PRNGSequence(1234)
        self.input_data = self.__setup_data()

    def __setup_data(self):
        node_feats = jrandom.normal(next(self.key), (node_num, in_feats))
        adj = jnp.array([
            [0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9],
            [4, 7, 3, 2, 1, 1, 8, 0, 8, 6, 5, 9, 0, 3, 5, 6],
        ])
        graph_idx = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        return (node_feats, adj, graph_idx, True)

    def __forward(self, node_feats, adj, graph_idx, is_training):
        model = SparseGCNPredicator(in_feats=in_feats, hidden_feats=hidden_feats,
                                    pooling_method=pooling_method,
                                    predicator_hidden_feats=predicator_hidden_feats, n_out=n_out)
        return model(node_feats, adj, graph_idx, is_training)

    def test_forward_shape(self):
        """Test output shape of SparseGCNPredicator"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (batch_size, n_out)
