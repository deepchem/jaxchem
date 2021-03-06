from typing import List, Optional

import haiku as hk
import jax
import jax.numpy as jnp


from jaxchem.models.readout.graph_pooling import sparse_graph_pooling
from jaxchem.models.gcn.sparse_pattern.gcn import SparseGCN
from jaxchem.typing import Activation, Pooling


class SparseGCNPredicator(hk.Module):
    """GCN Predicator is a wrapper function using GCN and MLP."""

    def __init__(self, in_feats: int, hidden_feats: List[int], activation: Optional[List[Activation]] = None,
                 batch_norm: Optional[List[bool]] = None, dropout: Optional[List[float]] = None,
                 pooling_method: Pooling = 'mean', predicator_hidden_feats: int = 128,
                 predicator_dropout: float = 0.0, n_out: int = 1,
                 name: Optional[str] = None):
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
        pooling_method : Literal['max', 'min', 'mean', 'sum']
            pooling method name, default to 'mean'.
        predicator_hidden_feats : int
            Size of hidden graph representations in the predicator, default to 128.
        predicator_dropout : float
            The probability for dropout in the predicator, default to 0.0.
        n_out : int
            Number of the output size, default to 1.
        """
        super(SparseGCNPredicator, self).__init__(name=name)
        self.gcn = SparseGCN(in_feats, hidden_feats, activation=activation,
                             batch_norm=batch_norm, dropout=dropout)
        self.pooling = sparse_graph_pooling(pooling_method)
        self.fc = hk.Linear(hidden_feats[-1])
        self.predicator_dropout = 0.0 if predicator_dropout is None else predicator_dropout
        self.activation = jax.nn.relu
        self.out = hk.Linear(n_out)

    def __call__(self, node_feats: jnp.ndarray, adj: jnp.ndarray, graph_idx: jnp.array,
                 is_training: bool) -> jnp.ndarray:
        """Predict logits or values

        Parameters
        ----------
        node_feats : ndarray of shape (N, in_feats)
            Batch input node features.
            N is the total number of nodes in the batch
        adj : ndarray of shape (2, E)
            Batch adjacency list.
            E is the total number of edges in the batch
        graph_idx : ndarray of shape (N,)
            This idx indicate a graph number for node_feats in the batch.
            When the two nodes shows the same graph idx, these belong to the same graph.
        is_training : bool
            Whether the model is training or not.

        Returns
        -------
        out : ndarray of shape (batch_size, n_out)
            Predicator output.
        """
        predicator_dropout = self.predicator_dropout if is_training is True else 0.0
        node_feats = self.gcn(node_feats, adj, is_training)
        # pooling
        graph_feat = self.pooling(node_feats, graph_idx)
        if predicator_dropout != 0.0:
            graph_feat = hk.dropout(hk.next_rng_key(), predicator_dropout, graph_feat)
        graph_feat = self.fc(graph_feat)
        graph_feat = self.activation(graph_feat)
        out = self.out(graph_feat)
        return out
