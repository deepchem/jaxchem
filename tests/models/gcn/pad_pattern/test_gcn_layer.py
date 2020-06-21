import unittest

import haiku as hk
import jax.numpy as jnp
import jax.random as jrandom
from jaxchem.models import PadGCNLayer


# params
in_feats = 64
out_feats = 32
batch_size = 32
max_node_size = 30


class TestPadGCNLayer(unittest.TestCase):
    """Test PadGCNLayer"""

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
        model = PadGCNLayer(in_feats=in_feats, out_feats=out_feats)
        return model(node_feats, adj, is_training)

    def __forward_with_batch_norm(self, node_feats, adj, is_training):
        model = PadGCNLayer(in_feats=in_feats, out_feats=out_feats, batch_norm=True)
        return model(node_feats, adj, is_training)

    def test_forward_shape(self):
        """Test output shape of PadGCNLayer"""
        forward = hk.transform_with_state(self.__forward)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (batch_size, max_node_size, out_feats)

    def test_forward_shape_with_batch_norm(self):
        """Test output shape of PadGCNLayer with BatchNorm"""
        forward = hk.transform_with_state(self.__forward_with_batch_norm)
        params, state = forward.init(next(self.key), *self.input_data)
        preds, _ = forward.apply(params, state, next(self.key), *self.input_data)
        assert preds.shape == (batch_size, max_node_size, out_feats)
