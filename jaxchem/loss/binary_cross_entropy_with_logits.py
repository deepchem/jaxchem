import jax.numpy as jnp


def binary_cross_entropy_with_logits(inputs: jnp.ndarray, targets: jnp.ndarray,
                                     average: bool = True) -> jnp.ndarray:
    """Binary cross entropy loss.

    This function is based on the PyTorch implemantation.

    See : https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246

    Parameters
    ----------
    inputs : jnp.ndarray
        This is a model output. This is a value before passing a sigmoid function.
    targets : jnp.ndarray
        This is a label and the same shape as inputs.
    average : bool
        Whether to mean loss values or sum, default to be True.

    Returns
    -------
    loss : jnp.ndarray
        This is a binary cross entropy loss.
    """

    if inputs.shape != targets.shape:
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            targets.shape, inputs.shape))

    max_val = jnp.clip(-inputs, 0, None)
    loss = inputs - inputs * targets + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-inputs - max_val)))

    if average:
        return jnp.mean(loss)

    return jnp.sum(loss)
