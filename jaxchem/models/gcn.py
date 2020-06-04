import jax.numpy as jnp
from jax import random
from jax.nn import relu
from jax.nn.initializers import he_normal, normal
from jax.experimental.stax import BatchNorm, Dropout


from jaxchem.utils import sparse_matmul


def GCNLayer(out_dim, activation=relu, bias=True, sparse=False,
             W_init=he_normal(), b_init=normal()):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    out_dim : int
        Number of output node features.
    activation : activation function
        Default to be relu function.
    bias : bool
        Whether to add bias after affine transformation, default to be True.
    sparse : bool
        Whether to use the matrix multiplication method for sparse matrix, default to be False.
    W_init : initialize function for weight
        Default to be He normal distribution.
    b_init : initialize function for bias
        Default to be normal distribution.
    """
    def _matmul(A, B, shape):
        if sparse:
            return sparse_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,)) if bias else None
        return output_shape, (W, b)

    def apply_fun(params, x, adj, **kwargs):
        W, b = params
        support = jnp.dot(x, W)
        out = _matmul(adj, support, support.shape[0])
        if bias:
            out += b
        out = activation(out)
        return out

    return init_fun, apply_fun


def GCN(hidden_feats, activation=None, batchnorm=None, dropout=None,
        bias=True, sparse=False):
    r"""GCN `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    hidden_feats : list[int]
        List of output node features.
    activation : list[activation function]
        Default to be relu function.
    batchnorm : list[bool] or None
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list[float] or None
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    bias : bool
        Whether to add bias after affine transformation, default to be True.
    sparse : bool
        Whether to use the matrix multiplication method for sparse matrix,  default to be False.
    """
    layer_num = len(hidden_feats)
    if activation is None:
        activation = [relu for _ in range(layer_num)]
    if batchnorm is None:
        batchnorm = [False for _ in range(len(layer_num))]
    if dropout is None:
        dropout = [0.0 for _ in range(len(layer_num))]

    lengths = [len(hidden_feats), len(activation),
               len(batchnorm), len(dropout)]
    assert len(set(lengths)) == 1, \
        'Expect the lengths of hidden_feats, activation, ' \
        'batchnorm and dropout to be the same, ' \
        'got {}'.format(lengths)

    # initialize layer
    gcn_init_funcs = [None for _ in range(layer_num)]
    gcn_funcs = [None for _ in range(layer_num)]
    batchnorm_init_funcs = [None for _ in range(layer_num)]
    batchnorm_funcs = [None for _ in range(layer_num)]
    drop_funcs = [None for _ in range(layer_num)]
    for i, (out_dim, act, bnorm, rate) in enumerate(zip(hidden_feats, activation, batchnorm, dropout)):
        gcn_init_funcs[i], gcn_funcs[i] = GCNLayer(out_dim, activation=act, bias=bias, sparse=sparse)
        if bnorm:
            batchnorm_init_funcs[i], batchnorm_funcs[i] = BatchNorm()
        if rate != 0.0:
            _, drop_func = Dropout(rate)
            drop_funcs[i] = drop_func

    def init_fun(rng, input_shape):
        gcn_params = [None for _ in range(layer_num)]
        batchnorm_prams = [None for _ in range(layer_num)]
        for i in range(layer_num):
            rng, gcn_rng = random.split(rng)
            input_shape, gcn_param = gcn_init_funcs[i](gcn_rng, input_shape)
            gcn_params[i] = gcn_param
            if batchnorm[i]:
                rng, batchnorm_rng = random.split(rng)
                input_shape, batchnorm_param = \
                    batchnorm_init_funcs[i](batchnorm_rng, input_shape)
                batchnorm_prams[i] = batchnorm_param
        return input_shape, (gcn_params, batchnorm_prams)

    def apply_fun(params, x, adj, mode='train', **kwargs):
        rng = kwargs.pop('rng', None)
        gcn_params, batchnorm_prams = params
        for i in range(layer_num):
            rng, k = random.split(rng)
            x = gcn_funcs[i](gcn_params[i], x, adj, rng=k)
            if batchnorm[i]:
                rng, k = random.split(rng)
                x = batchnorm_funcs[i](batchnorm_prams[i], x, rng=k)
            if dropout[i] != 0.0:
                rng, k = random.split(rng)
                x = drop_funcs[i](x, mode, rng=k)
        return x

    return init_fun, apply_fun
