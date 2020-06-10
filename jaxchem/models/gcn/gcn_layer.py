import jax.numpy as jnp
from jax import random, vmap
from jax.nn import relu
from jax.nn.initializers import he_normal, normal
from jax.experimental.stax import BatchNorm


from jaxchem.models.nn.dropout import Dropout


def GCNLayer(out_dim, activation=relu, bias=True, sparse=False,
             batch_norm=False, dropout=0.0, W_init=he_normal(), b_init=normal()):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`

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
        The probability for dropout, default to 0.0.
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
        """Initialize parameters.

        Parameters
        ----------
        rng : PRNGKey
            rng is a value for generating random values.
        input_shape : (batch_size, N, M1)
            The shape of input (input node features).
            N is the total number of nodes in the batch of graphs.
            M1 is the input node feature size.

        Returns
        -------
        output_shape : (batch_size, N, M2)
            The shape of output (new node features).
            M2 is the new node feature size and equal to out_dim.
        params: Tuple (W, b, batch_norm_param)
            W is a weight and b is a bias.
            W : ndarray of shape (N, M2) or None
            b : ndarray of shape (M2,)
            batch_norm_param : Tuple (beta, gamma) or None
        """
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3 = random.split(rng, 3)
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,)) if bias else None
        batch_norm_param = None
        if batch_norm:
            output_shape, batch_norm_param = batch_norm_init(k3, output_shape)
        return output_shape, (W, b, batch_norm_param)

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
            rng is a value for generating random values
        is_train : bool
            Whether the model is training or not.

        Returns
        -------
        new_node_feats : ndarray of shape (batch_size, N, M2)
            Batched new node features.
            M2 is the new node feature size and equal to out_dim.
        """
        W, b, batch_norm_param = params

        # H' = A × H × W
        def node_update_func(node_feats, adj):
            return jnp.matmul(adj, jnp.dot(node_feats, W))
        # batched operation for updating node features
        new_node_feats = vmap(node_update_func)(node_feats, adj)

        if bias:
            new_node_feats += b
        new_node_feats = activation(new_node_feats)
        if dropout != 0.0:
            rng, key = random.split(rng)
            new_node_feats = drop_fun(None, new_node_feats, is_train, rng=key)
        if batch_norm:
            new_node_feats = batch_norm_fun(batch_norm_param, new_node_feats)
        return new_node_feats

    return init_fun, apply_fun
