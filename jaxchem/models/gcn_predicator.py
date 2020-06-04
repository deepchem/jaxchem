import jax.numpy as jnp
from jax import random
from jax.experimental.stax import Dense, Dropout, Relu, serial


from jaxchem.models import GCN


def GCNPredicator(hidden_feats, activation=None, batchnorm=None, dropout=None,
                  classifier_hidden_feats=64, classifier_dropout=None,
                  n_tasks=1, bias=True, sparse=False):
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
    classifier_hidden_feats : int
        Size of hidden graph representations in the classifier. Default to 128.
    classifier_dropout : float
        The probability for dropout in the classifier. Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    bias : bool
        Whether to add bias after affine transformation, default to be True.
    sparse : bool
        Whether to use the matrix multiplication method for sparse matrix,  default to be False.
    """

    gcn_init, gcn_func = GCN(hidden_feats, activation, batchnorm, dropout, bias, sparse)
    classifier_dropout = 0.0 if classifier_dropout is None else classifier_dropout
    _, drop_fun = Dropout(classifier_dropout)
    dnn_init, dnn_func = serial(
        Dense(classifier_hidden_feats), Relu,
        Dense(n_tasks),
    )

    def init_fun(rng, input_shape):
        rng, gcn_rng, dnn_rng = random.split(3)
        input_shape, gcn_param = gcn_init(gcn_rng, input_shape)
        input_shape, dnn_param = dnn_init(dnn_rng, input_shape)
        return input_shape, (gcn_param, dnn_param)

    def apply_fun(params, x, adj, mode='train', **kwargs):
        rng = kwargs.pop('rng', None)
        gcn_param, dnn_param = params
        rng, gcn_rng, dnn_rng, dropout_rng = random.split(4)
        x = gcn_func(gcn_param, x, adj, rng=gcn_rng)
        # mean pooling
        x = jnp.mean(x, axis=0)
        if classifier_dropout != 0.0:
            x = drop_fun(x, mode, rng=dropout_rng)
        x = dnn_func(dnn_param, x, rng=dnn_rng)
        return x

    return init_fun, apply_fun
