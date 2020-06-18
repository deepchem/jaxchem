import jax.numpy as jnp
from jax import nn


def clipped_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Customize sigmoid function to avoid an overflow.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x : ndarray
    """
    # x is clipped because nn.sigmoid sometimes get overflow and return nan
    # restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
    sigmoid_range = 34.538776394910684
    x = jnp.clip(x, -sigmoid_range, sigmoid_range)
    x = nn.sigmoid(x)
    return x
