from typing import Callable

import jax.numpy as jnp
from jax.ops import index_add, index_min, index_max


from jaxchem.typing import Pooling


def pad_graph_pooling(method: Pooling = 'mean') -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Pooling function for pad pattern graph data.

    method : str
        pooling method name.

    Returns
    -------
    Function
        This function aggregates node_feats about axis=1.
    """
    if method == 'mean':
        return lambda node_feats: jnp.mean(node_feats, axis=1)
    elif method == 'sum':
        # FIXME : When using np.sum, Nan happens...
        return lambda node_feats: jnp.sum(node_feats, axis=1)
    elif method == 'max':
        return lambda node_feats: jnp.max(node_feats, axis=1)
    elif method == 'min':
        return lambda node_feats: jnp.min(node_feats, axis=1)
    else:
        raise ValueError("{} is an unsupported pooling method. \
            Currently, you can only use mean, sum, max and min pooling.")


def sparse_graph_pooling(method: Pooling = 'mean') -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Pooling function for sparse pattern graph data.

    method : str
        pooling method name.

    Returns
    -------
    Function
        This function aggregates node_feats with graph_idx.
    """
    if method == 'mean':
        return lambda node_feats, graph_idx: sparse_mean_pooling(node_feats, graph_idx)
    elif method == 'sum':
        return lambda node_feats, graph_idx: sparse_sum_pooling(node_feats, graph_idx)
    elif method == 'max':
        return lambda node_feats, graph_idx: sparse_max_pooling(node_feats, graph_idx)
    elif method == 'min':
        return lambda node_feats, graph_idx: sparse_min_pooling(node_feats, graph_idx)
    else:
        raise ValueError("{} is an unsupported pooling method. \
            Currently, you can only use mean, sum, max and min pooling.")


def sparse_mean_pooling(node_feats: jnp.ndarray, graph_idx: jnp.ndarray) -> jnp.ndarray:
    """Mean pooling function for sparse pattern graph data.

    node_feats : ndarray of shape (N, in_feats)
        Batch input node features.
        N is the total number of nodes in the batch.
    graph_idx : ndarray of shape (N,)
        This idx indicate a graph number for node_feats in the batch.
        When the two nodes shows the same graph idx, these belong to the same graph.

    Returns
    -------
    ndarray of shape (batch_size, in_feats)
        Batch graph features.
    """
    _, feat_dim = node_feats.shape
    batch_size = graph_idx[-1] + 1
    n_atom = index_add(jnp.zeros(batch_size), graph_idx, 1).reshape(-1, 1)
    init_matrix = jnp.zeros((batch_size, feat_dim))
    return index_add(init_matrix, graph_idx, node_feats) / n_atom


def sparse_sum_pooling(node_feats: jnp.ndarray, graph_idx: jnp.ndarray) -> jnp.ndarray:
    """Sum pooling function for sparse pattern graph data.

    node_feats : ndarray of shape (N, in_feats)
        Batch input node features.
        N is the total number of nodes in the batch.
    graph_idx : ndarray of shape (N,)
        This idx indicate a graph number for node_feats in the batch.
        When the two nodes shows the same graph idx, these belong to the same graph.

    Returns
    -------
    ndarray of shape (batch_size, in_feats)
        Batch graph features.
    """
    _, feat_dim = node_feats.shape
    batch_size = graph_idx[-1] + 1
    init_matrix = jnp.zeros((batch_size, feat_dim))
    return index_add(init_matrix, graph_idx, node_feats)


def sparse_max_pooling(node_feats: jnp.ndarray, graph_idx: jnp.ndarray) -> jnp.ndarray:
    """Max pooling function for sparse pattern graph data.

    node_feats : ndarray of shape (N, in_feats)
        Batch input node features.
        N is the total number of nodes in the batch
    graph_idx : ndarray of shape (N,)
        This idx indicate a graph number for node_feats in the batch.
        When the two nodes shows the same graph idx, these belong to the same graph.

    Returns
    -------
    ndarray of shape (batch_size, in_feats)
        Batch graph features.
    """
    _, feat_dim = node_feats.shape
    batch_size = graph_idx[-1] + 1
    init_matrix = jnp.ones((batch_size, feat_dim)) * -jnp.inf
    return index_max(init_matrix, graph_idx, node_feats)


def sparse_min_pooling(node_feats: jnp.ndarray, graph_idx: jnp.ndarray) -> jnp.ndarray:
    """Min pooling function for sparse pattern graph data.

    node_feats : ndarray of shape (N, in_feats)
        Batch input node features.
        N is the total number of nodes in the batch
    graph_idx : ndarray of shape (N,)
        This idx indicate a graph number for node_feats in the batch.
        When the two nodes shows the same graph idx, these belong to the same graph.

    Returns
    -------
    ndarray of shape (batch_size, in_feats)
        Batch graph features.
    """
    _, feat_dim = node_feats.shape
    batch_size = graph_idx[-1] + 1
    init_matrix = jnp.ones((batch_size, feat_dim)) * jnp.inf
    return index_min(init_matrix, graph_idx, node_feats)
