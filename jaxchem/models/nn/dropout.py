import jax.numpy as jnp
from jax import random


def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental. Dropout, to use
    `is_train` as an argument to apply_fun, instead of defining it at
    definition time.

    Parameters
    ----------
    rate : float between 0 and 1
        Fraction of the input units to drop.

    Returns
    -------
    init_fun : Function
        Initializes the parameters of the layer.
    apply_fun : Function
        Defines the forward computation function.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, is_train, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        if is_train is True:
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs
    return init_fun, apply_fun
