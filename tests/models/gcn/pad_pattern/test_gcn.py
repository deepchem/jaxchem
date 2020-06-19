import unittest

import haiku as hk
import jax.random as jrandom
from jaxchem.models import PadGCN


# params
in_feats = 64
hidden_feats = [64, 32, 16]
batch_size = 32
max_node_size = 30


class TestPadGCN(unittest.TestCase):
    """Test PadGCN"""

    def setup_method(self, method):
        self.key = hk.PRNGSequence(1234)
        self.input_data = self.__setup_data()

    def __setup_data(self):
        batched_node_feats = jrandom.normal(next(self.key), (batch_size, max_node_size, in_feats))
        batched_adj = jrandom.normal(next(self.key), (batch_size, max_node_size, max_node_size))
        return (batched_node_feats, batched_adj, True)

    def __forward(self, node_feats, adj, is_training):
        model = PadGCN(in_feats=in_feats, hidden_feats=hidden_feats)
        return model(node_feats, adj, is_training)

    def test_forward_shape(self):
        """Test output shape of PadGCN"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (batch_size, max_node_size, hidden_feats[-1])
