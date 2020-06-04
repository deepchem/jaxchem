"""
Original code from jax-gnn by @gcucurull
https://github.com/gcucurull/jax-gcn/blob/master/sparse_matmul.py
"""

import jax


@jax.partial(jax.jit, static_argnums=(2))
def sparse_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res
