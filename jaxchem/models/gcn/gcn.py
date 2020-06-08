import jax.numpy as jnp
from jax import random, vmap
from jax.nn import relu
from jax.nn.initializers import he_normal, normal
from jax.experimental.stax import BatchNorm


from jaxchem.models import Dropout


def GCNLayer(out_dim, activation=relu, bias=True, sparse=False,
             batch_norm=False, dropout=0.0, W_init=he_normal(), b_init=normal()):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    out_dim : int
        Number of output node features.
    activation : Function
        activation function, default to be relu function.
    bias : bool
        Whether to add bias after affine transformation, default to be True.
    sparse : bool
        Whether to use the matrix multiplication method for sparse matrix, default to be False.
    batch_norm : bool
        Whetehr to use BatchNormalization or not, default to be False.
    dropout : float
        Whetehr to use BatchNormalization or not, default to be False.
    W_init : initialize function for weight
        Default to be He normal distribution.
    b_init : initialize function for bias
        Default to be normal distribution.

    Returns
    -------
    init_fun : Function
        Initializes the parameters of the layer.
    apply_fun : Function
        Defines the forward computation function.
    """

    _, drop_fun = Dropout(dropout)
    batch_norm_init, batch_norm_fun = BatchNorm()

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3 = random.split(rng, 3)
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,)) if bias else None
        batch_norm_param = None
        if batch_norm:
            output_shape, batch_norm_param = batch_norm_init(k3, output_shape)
        return output_shape, (W, b, batch_norm_param)

    def apply_fun(params, node_feats, adj, rng, is_train=True):
        W, b, batch_norm_param = params

        # H' = A × H × W
        def node_update_func(node_feats, adj):
            return jnp.matmul(adj, jnp.dot(node_feats, W))
        # batched operation for updating node features
        out = vmap(node_update_func)(node_feats, adj)

        if bias:
            out += b
        out = activation(out)
        if dropout != 0.0:
            rng, k = random.split(rng)
            out = drop_fun(None, out, is_train, rng=k)
        if batch_norm:
            out = batch_norm_fun(batch_norm_param, out)
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
    activation : list[Function] or None
        ``activation[i]`` is the activation function of the i-th GCN layer.
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
        Whether to use the matrix multiplication method for sparse matrix, default to be False.

    Returns
    -------
    init_fun : Function
        Initializes the parameters of the layer.
    apply_fun : Function
        Defines the forward computation function.
    """
    layer_num = len(hidden_feats)
    if activation is None:
        activation = [relu for _ in range(layer_num)]
    if batchnorm is None:
        batchnorm = [False for _ in range(layer_num)]
    if dropout is None:
        dropout = [0.0 for _ in range(layer_num)]

    lengths = [len(hidden_feats), len(activation),
               len(batchnorm), len(dropout)]
    assert len(set(lengths)) == 1, \
        'Expect the lengths of hidden_feats, activation, ' \
        'batchnorm and dropout to be the same, ' \
        'got {}'.format(lengths)

    # initialize layer
    gcn_init_funs = [None for _ in range(layer_num)]
    gcn_funs = [None for _ in range(layer_num)]
    for i, (out_dim, act, bnorm, rate) in enumerate(zip(hidden_feats, activation, batchnorm, dropout)):
        gcn_init_funs[i], gcn_funs[i] = \
            GCNLayer(out_dim, activation=act, bias=bias, sparse=sparse,
                     batch_norm=bnorm, dropout=rate)

    def init_fun(rng, input_shape):
        gcn_params = [None for _ in range(layer_num)]
        for i in range(layer_num):
            rng, gcn_rng = random.split(rng)
            input_shape, gcn_param = gcn_init_funs[i](gcn_rng, input_shape)
            gcn_params[i] = gcn_param
        return input_shape, gcn_params

    def apply_fun(params, node_feats, adj, rng, is_train=True):
        for i in range(layer_num):
            rng, k = random.split(rng)
            node_feats = gcn_funs[i](params[i], node_feats, adj, k, is_train)
        return node_feats

    return init_fun, apply_fun
