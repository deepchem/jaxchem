import argparse
import time

import itertools

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, nn, random, vmap
from jax.experimental import optimizers

from jaxchem.models import GCNPredicator
from deepchem.molnet import load_tox21


task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
              'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


def parse_arguments():
    parser = argparse.ArgumentParser('Tox21 example')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--task', type=str, choices=task_names, default='NR-AR')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # load tox21 dataset
    tox21_tasks, tox21_datasets, _ = load_tox21(featurizer='AdjacencyConv', reload=True)
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    # define hyperparams
    rng = random.PRNGKey(args.seed)
    # model params
    hidden_feats = [64, 64, 64]
    activation, batchnorm, dropout = None, None, None  # use default
    predicator_hidden_feats = 32
    predicator_dropout = None  # use default
    n_out = 1  # binary classification
    # training params
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    task = args.task

    # setup model
    init_fun, predict_fun = \
        GCNPredicator(hidden_feats=hidden_feats, activation=activation, batchnorm=batchnorm,
                      dropout=dropout, predicator_hidden_feats=predicator_hidden_feats,
                      predicator_dropout=predicator_dropout, n_out=n_out)
    # make a batched version of the `predict` function
    batched_predict = vmap(predict_fun, in_axes=(None, 0, 1))

    # init params
    rng, init_key = random.split(rng)
    sample_node_feat = train_dataset.X[0][1]
    input_shape = sample_node_feat.shape
    _, init_params = init_fun(init_key, input_shape)
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    opt_state = opt_init(init_params)

    # no batch data is ok!
    adj, node_feat = train_dataset.X[0]
    preds = predict_fun(init_params, node_feat, adj, rng, True)
    assert preds.shape == (1,)

    # define training loss (cross entropy)
    @jit
    def loss(params, batch):
        """Compute the loss of the network"""
        preds = batched_predict(params, *batch)
        logits = nn.sigmoid(preds)
        return -(targets * jnp.log(logits) + (1 - targets) * (1 - jnp.log(logits)))

    # define training update
    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    # define evaluate metric
    @jit
    def accuracy(params, batch):
        preds = batched_predict(params, *batch)
        logits = nn.sigmoid(preds)
        predicted_class = jnp.where(logits < 0.5, 0, 1)
        return jnp.mean(predicted_class == targets)

    print("Starting training...")
    task_index = tox21_tasks.index(task)
    itercount = itertools.count()
    for epoch in range(num_epochs):
        for original_batch in train_dataset.iterbatches(batch_size=batch_size):
            # This is needed for getting batched node_feats and adj matrix...
            inputs, targets, _, _ = original_batch
            node_feats = np.array([inputs[i][1] for i in range(len(inputs))])
            adj = np.array([inputs[i][0] for i in range(len(inputs))])
            targets = targets[:, task_index]
            rng, key = random.split(rng)
            batch = (node_feats, adj, targets, key, True)
            # update
            # FIXME : I struggle the error.... (maybe related to vmap)
            opt_state = update(next(itercount), opt_state, batch)

    return


if __name__ == "__main__":
    main()
