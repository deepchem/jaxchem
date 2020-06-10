import jax.numpy as jnp
from jax import random
from jax.experimental.stax import Dense, Relu, serial


from jaxchem.models.gcn.gcn import GCN
from jaxchem.models.nn.dropout import Dropout


def GCNPredicator(hidden_feats, activation=None, batchnorm=None, dropout=None,
                  predicator_hidden_feats=64, predicator_dropout=None,
                  n_out=1, bias=True, sparse=False):
    r"""GCN Predicator is a wrapper function using GCN and MLP.

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
    predicator_dropout : float or None
        The probability for dropout in the predicator, default to None.
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
        """Initialize parameters.

        Parameters
        ----------
        rng : PRNGKey
        input_shape :  (batch_size, N, M1)
            The shape of input (input node features).
            N is the total number of nodes in the batch of graphs.
            M1 is the input node feature size.

        Returns
        -------
        output_shape : (batch_size, M4)
            The shape of output.
            M4 is the output size of GCNPredicator and equal to n_out.
        params: Tuple (gcn_param, dnn_param)
            gcn_param is all parameters of GCN.
            dnn_param is all parameters of full connected layer.
        """
        output_shape = input_shape
        rng, gcn_rng, dnn_rng = random.split(rng, 3)
        output_shape, gcn_param = gcn_init(gcn_rng, output_shape)
        output_shape, dnn_param = dnn_init(dnn_rng, output_shape)
        return output_shape, (gcn_param, dnn_param)

    def apply_fun(params, node_feats, adj, rng, is_train):
        """Define forward computation function.

        Parameters
        ----------
        node_feats : ndarray of shape (batch_size, N, M1)
            Batched input node features.
            N is the total number of nodes in the batch of graphs.
            M1 is the input node feature size.
        adj : ndarray of shape (batch_size, N, N)
            Batched adjacency matrix.
        rng : PRNGKey
            rng is a value for generating random values
        is_train : bool
            Whether the model is training or not.

        Returns
        -------
        out : ndarray of shape (batch_size, M4)
            The shape of output.
            M4 is the output size of GCNPredicator and equal to n_out.
        """
        gcn_param, dnn_param = params
        rng, gcn_rng, dropout_rng = random.split(rng, 3)
        node_feats = gcn_fun(gcn_param, node_feats, adj, gcn_rng, is_train)
        # mean pooling
        graph_feat = jnp.mean(node_feats, axis=1)
        if predicator_dropout != 0.0:
            graph_feat = drop_fun(None, graph_feat, is_train, rng=dropout_rng)
        out = dnn_fun(dnn_param, graph_feat)
        return out

    return init_fun, apply_fun
