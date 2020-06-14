import unittest

import jax.random as jrandom
from jaxchem.models import GCNLayer


# params
OUT_DIM = 32
BATCH_SIZE = 32
MAX_NODE_SIZE = 30
NODE_FEATURE_DIM = 64


class TestGCNlayer(unittest.TestCase):
    """Test GCNLayer"""

    def setup_method(self, method):
        self.key = jrandom.PRNGKey(1234)
        self.input_data = self.__setup_data()
        self.models_fun = GCNLayer(out_dim=OUT_DIM)

    def __setup_data(self):
        self.key, k1, k2, k3 = jrandom.split(self.key, 4)
        batched_node_feats = jrandom.normal(k1, (BATCH_SIZE, MAX_NODE_SIZE, NODE_FEATURE_DIM))
        batched_adj = jrandom.normal(k2, (BATCH_SIZE, MAX_NODE_SIZE, MAX_NODE_SIZE))
        return (batched_node_feats, batched_adj, k3, True)

    def test_forward_shape(self):
        """Test output shape of GCNLayer"""
        init_fun, predict_fun = self.models_fun
        out_shape, params = init_fun(self.key, self.input_data[0].shape)
        preds = predict_fun(params, *self.input_data)
        assert preds.shape == out_shape
        assert preds.shape == (BATCH_SIZE, MAX_NODE_SIZE, OUT_DIM)
