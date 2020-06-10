from jax import random
from jax.nn import relu


from jaxchem.models.gcn.gcn_layer import GCNLayer


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
        """Initialize parameters.

        Parameters
        ----------
        rng : PRNGKey
            rng is a value for generating random values.
        input_shape :  (batch_size, N, M1)
            The shape of input (input node features).
            N is the total number of nodes in the batch of graphs.
            M1 is the input node feature size.

        Returns
        -------
        output_shape : (batch_size, N, M3)
            The shape of output (new node features).
            M3 is the new node feature size and equal to hidden_feats[-1].
        gcn_params: list[params of GCNLayer]
            All parameter of GCN layer.
        """
        gcn_params = [None for _ in range(layer_num)]
        output_shape = input_shape
        for i in range(layer_num):
            rng, gcn_rng = random.split(rng)
            output_shape, gcn_param = gcn_init_funs[i](gcn_rng, output_shape)
            gcn_params[i] = gcn_param
        return output_shape, gcn_params

    def apply_fun(params, node_feats, adj, rng, is_train):
        """Update node representations.

        Parameters
        ----------
        node_feats : ndarray of shape (batch_size, N, M1)
            Batched input node features.
            N is the total number of nodes in the batch of graphs.
            M1 is the input node feature size.
        adj : ndarray of shape (batch_size, N, N)
            Batched adjacency matrix.
        rng : PRNGKey
            rng is a value for generating random values.
        is_train : bool
            Whether the model is training or not.

        Returns
        -------
        new_node_feats : ndarray of shape (batch_size, N, M3)
            Batched new node features.
            M3 is the new node feature size and equal to hidden_feats[-1].
        """
        for i in range(layer_num):
            rng, key = random.split(rng)
            node_feats = gcn_funs[i](params[i], node_feats, adj, key, is_train)
        return node_feats

    return init_fun, apply_fun
