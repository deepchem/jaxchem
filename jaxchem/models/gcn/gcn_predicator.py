import jax.numpy as jnp
from jax import random
from jax.experimental.stax import Dense, Relu, serial


from jaxchem.models import GCN, Dropout


def GCNPredicator(hidden_feats, activation=None, batchnorm=None, dropout=None,
                  predicator_hidden_feats=64, predicator_dropout=None,
                  n_out=1, bias=True, sparse=False):
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
    predicator_hidden_feats : int
        Size of hidden graph representations in the predicator, default to 128.
    predicator_dropout : float
        The probability for dropout in the predicator, default to 0.
    n_out : int
        Number of the output size, default to 1.
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
    gcn_init, gcn_fun = GCN(hidden_feats, activation, batchnorm, dropout, bias, sparse)
    predicator_dropout = 0.0 if predicator_dropout is None else predicator_dropout
    _, drop_fun = Dropout(predicator_dropout)
    dnn_layers = [Dense(predicator_hidden_feats), Relu, Dense(n_out)]
    dnn_init, dnn_fun = serial(*dnn_layers)

    def init_fun(rng, input_shape):
        rng, gcn_rng, dnn_rng = random.split(rng, 3)
        input_shape, gcn_param = gcn_init(gcn_rng, input_shape)
        input_shape, dnn_param = dnn_init(dnn_rng, input_shape)
        return input_shape, (gcn_param, dnn_param)

    def apply_fun(params, node_feats, adj, rng, is_train=True):
        gcn_param, dnn_param = params
        rng, gcn_rng, dnn_rng, dropout_rng = random.split(rng, 4)
        node_feats = gcn_fun(gcn_param, node_feats, adj, gcn_rng, is_train)
        # mean pooling
        graph_feat = jnp.mean(node_feats, axis=1)
        if predicator_dropout != 0.0:
            graph_feat = drop_fun(None, graph_feat, is_train, rng=dropout_rng)
        out = dnn_fun(dnn_param, graph_feat)
        return out

    return init_fun, apply_fun
