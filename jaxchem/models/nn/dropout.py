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
        """Update node representations.

        Parameters
        ----------
        params : None
            There is no params for Dropout
        inputs : ndarray of shape
            The input for Dropout.
        is_train : bool
            Whether the model is training or not.
        rng : PRNGKey
            rng is a value for generating random values. Dropout must require rng.

        Returns
        -------
        out : ndarray of shape
            The output for Dropout. The shape is equal to the shape of inputs
        """
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        if is_train is True:
            keep_rate = 1 - rate
            keep = random.bernoulli(rng, keep_rate, inputs.shape)
            return jnp.where(keep, inputs / keep_rate, 0)
        else:
            return inputs
    return init_fun, apply_fun
