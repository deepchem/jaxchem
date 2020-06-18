from typing import Callable
from typing_extensions import Literal
import jax.numpy as jnp

Pooling = Literal['max', 'min', 'mean', 'sum']
Activation = Callable[[jnp.ndarray], jnp.ndarray]
