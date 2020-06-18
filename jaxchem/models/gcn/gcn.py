from typing import List, Optional

import haiku as hk
import jax
import jax.numpy as jnp


from jaxchem.models.gcn.gcn_layer import GCNLayer
from jaxchem.typing import Activation


class GCN(hk.Module):
    """GCN `Semi-Supervised Classification with Graph Convolutional Networks`
        ref : <https://arxiv.org/abs/1609.02907>
    """

    def __init__(self, in_feats: int, hidden_feats: List[int], activation: Optional[List[Activation]] = None,
                 batch_norm: Optional[List[bool]] = None, dropout: Optional[List[float]] = None,
                 bias: bool = None, normalize: bool = True, name: Optional[str] = None):
        """Initializes the module.

        Parameters
        ----------
        in_feats : int
            Number of input node features.
        hidden_feats : list[int]
            List of output node features.
        activation : list[Activation] or None
            ``activation[i]`` is the activation function of the i-th GCN layer.
            ``len(activation)`` equals the number of GCN layers. By default,
            the activation each layer is relu function.
        batch_norm : list[bool] or None
            ``batch_norm[i]`` decides if batch normalization is to be applied on the output of
            the i-th GCN layer. ``len(batch_norm)`` equals the number of GCN layers. By default,
            batch normalization is not applied for all GCN layers.
        dropout : list[float] or None
            ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
            ``len(dropout)`` equals the number of GCN layers. By default, dropout is not
            performed for all layers.
        bias : bool
            Whether to add bias after affine transformation, default to be True.
        normalize : bool
            Whether to normalize the adjacency matrix or not, default to be True.
        """
        super(GCN, self).__init__(name=name)
        layer_num = len(hidden_feats)
        input_feats = [in_feats] + hidden_feats[:-1]
        out_feats = hidden_feats
        activation = activation or [jax.nn.relu for _ in range(layer_num)]
        batch_norm = batch_norm or [False for _ in range(layer_num)]
        dropout = dropout or [0.0 for _ in range(layer_num)]
        self.layer_num = layer_num
        self.layers = [
            GCNLayer(input_feats[i], out_feats[i], activation=activation[i],
                     batch_norm=batch_norm[i], dropout=dropout[i]) for i in range(layer_num)
        ]

        lengths = [len(input_feats), len(out_feats), len(activation),
                   len(batch_norm), len(dropout)]
        assert len(set(lengths)) == 1, \
            'Expect the lengths of hidden_feats, activation, ' \
            'batchnorm and dropout to be the same, ' \
            'got {}'.format(lengths)

    def __call__(self, node_feats: jnp.ndarray, adj: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        """Update node features.

        Parameters
        ----------
        node_feats : ndarray of shape (batch_size, N, in_feats)
            Batch input node features.
            N is the total number of nodes in the batch of graphs.
        adj : ndarray of shape (batch_size, N, N)
            Batch adjacency matrix.
        is_training : bool
            Whether the model is training or not.

        Returns
        -------
        new_node_feats : ndarray of shape (batch_size, N, out_feats)
            Batch new node features.
        """
        new_node_feats = node_feats
        for i in range(self.layer_num):
            new_node_feats = self.layers[i](new_node_feats, adj, is_training)
        return new_node_feats
