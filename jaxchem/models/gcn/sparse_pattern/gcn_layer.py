from typing import Optional

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from jax.ops import index_add


from jaxchem.typing import Activation


class SparseGCNLayer(hk.Module):
    """Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks`
        ref : <https://arxiv.org/abs/1609.02907>
    """

    def __init__(self, in_feats: int, out_feats: int, activation: Optional[Activation] = None,
                 bias: bool = True, normalize: bool = True, batch_norm: bool = False,
                 dropout: float = 0.0, w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None, name: Optional[str] = None):
        """Initializes the module.

        Parameters
        ----------
        in_feats : int
            Number of input node features.
        out_feats : int
            Number of output node features.
        activation : Activation or None
            activation function, default to be relu function.
        bias : bool
            Whether to add bias after affine transformation, default to be True.
        normalize : bool
            Whether to normalize or not, default to be True.
        batch_norm : bool
            Whetehr to use BatchNormalization or not, default to be False.
        dropout : float
            The probability for dropout, default to 0.0.
        W_init : initialize function for weight
            Default to be He truncated normal distribution.
        b_init : initialize function for bias
            Default to be truncated normal distribution.
        """
        super(SparseGCNLayer, self).__init__(name=name)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation or jax.nn.relu
        self.bias = bias
        self.normalize = normalize
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.w_init = w_init or hk.initializers.TruncatedNormal(np.sqrt(2. / in_feats))
        self.b_init = b_init or hk.initializers.TruncatedNormal()
        self.w = hk.get_parameter("w", shape=[in_feats, out_feats], init=self.w_init)
        self.b = hk.get_parameter("b", shape=[out_feats], init=self.b_init)

    def __call__(self, node_feats: jnp.ndarray, adj: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        """Update node features.

        Parameters
        ----------
        node_feats : ndarray of shape (N, in_feats)
            Batch input node features.
            N is the total number of nodes in the batch
        adj : ndarray of shape (2, E)
            Batch adjacency list.
            E is the total number of edges in the batch
        is_training : bool
            Whether the model is training or not.

        Returns
        -------
        new_node_feats : ndarray of shape (N, out_feats)
            Batch new node features.
        """
        dropout = self.dropout if is_training is True else 0.0

        # affine transformation
        new_node_feats = jnp.dot(node_feats, self.w)

        # aggregate
        if self.normalize:
            src_idx, dest_idx = adj[0], adj[1]
            degree = index_add(jnp.zeros(len(new_node_feats)), src_idx, 1).reshape(-1, 1)
            new_node_feats = new_node_feats / jnp.sqrt(degree)
            new_node_feats = index_add(new_node_feats, [dest_idx], new_node_feats[src_idx])
            new_node_feats = new_node_feats / jnp.sqrt(degree)
        else:
            src_idx, dest_idx = adj[0], adj[1]
            new_node_feats = index_add(new_node_feats, [dest_idx], new_node_feats[src_idx])

        if self.bias:
            new_node_feats += self.b
        new_node_feats = self.activation(new_node_feats)

        if dropout != 0.0:
            new_node_feats = hk.dropout(hk.next_rng_key(), dropout, new_node_feats)
        if self.batch_norm:
            new_node_feats = hk.BatchNorm(True, True, 0.9)(new_node_feats, is_training)

        return new_node_feats
