import unittest

import haiku as hk
import jax.numpy as jnp
import jax.random as jrandom
from jaxchem.models import PadGCNPredicator


# params
in_feats = 64
hidden_feats = [64, 32, 16]
pooling_method = 'mean'
predicator_hidden_feats = 16
n_out = 1
batch_size = 32
max_node_size = 30


class TestGCNPredicator(unittest.TestCase):
    """Test PadGCNPredicator"""

    def setup_method(self, method):
        self.key = hk.PRNGSequence(1234)
        self.input_data = self.__setup_data()

    def __setup_data(self):
        batched_node_feats = jrandom.normal(next(self.key), (batch_size, max_node_size, in_feats))
        batched_adj = jnp.where(
            jrandom.normal(next(self.key), (batch_size, max_node_size, max_node_size)) > 0,
            0, 1
        )
        return (batched_node_feats, batched_adj, True)

    def __forward(self, node_feats, adj, is_training):
        model = PadGCNPredicator(in_feats=in_feats, hidden_feats=hidden_feats,
                                 pooling_method=pooling_method,
                                 predicator_hidden_feats=predicator_hidden_feats, n_out=n_out)
        return model(node_feats, adj, is_training)

    def test_forward_shape(self):
        """Test output shape of PadGCNPredicator"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (batch_size, n_out)
