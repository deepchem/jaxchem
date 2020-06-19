from typing import Callable

import jax.numpy as jnp


from jaxchem.typing import Pooling


def pad_graph_pooling(method: Pooling = 'mean') -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Pooling function for pad pattern graph data.

    method : str
        pooling method name

    Returns
    -------
    Function
        This function aggregates node_feats about axis=1
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
