import os
import time
import random
import pickle
import argparse
from typing import Any, Tuple

import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix
from sklearn.metrics import roc_auc_score


from deepchem.molnet import load_tox21
from jaxchem.models import GCNPredicator, clipped_sigmoid
from jaxchem.utils import EarlyStopping


# type definition
PRNGKey = jnp.ndarray
Batch = Tuple[np.ndarray, np.ndarray, bool, np.ndarray]
OptState = Any

# task
task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
              'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


def parse_arguments():
    parser = argparse.ArgumentParser('Tox21 example')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, choices=task_names, default='NR-AR')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early-stop', type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def collate_fn(original_batch: Any, task_index: int, is_train: bool) -> Batch:
    """Make a correct batch as GCN model inputs"""
    # convert a batch returned by iterbatches to a correct batch as model inputs
    inputs, targets, _, _ = original_batch
    node_feats = np.array([inputs[i][1] for i in range(len(inputs))])
    adj = np.array([inputs[i][0] for i in range(len(inputs))])
    targets = targets[:, task_index]
    return (node_feats, adj, is_train, targets)


def main():
    args = parse_arguments()
    # fix seed
    seed_everything(args.seed)

    # load tox21 dataset
    tox21_tasks, tox21_datasets, _ = load_tox21(featurizer='AdjacencyConv', reload=True)
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    # define hyperparams
    rng_seq = hk.PRNGSequence(args.seed)
    # model params
    in_feats = train_dataset.X[0][1].shape[1]
    hidden_feats = [64, 64, 32]
    activation, batch_norm, dropout = None, None, None  # use default
    predicator_hidden_feats = 32
    pooling_method = 'mean'
    predicator_dropout = None  # use default
    n_out = 1  # binary classification
    # training params
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    task = args.task
    early_stop_patience = args.early_stop

    # setup model
    def forward(node_feats: jnp.ndarray, adj: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        model = GCNPredicator(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation,
                              batch_norm=batch_norm, dropout=dropout, pooling_method=pooling_method,
                              predicator_hidden_feats=predicator_hidden_feats,
                              predicator_dropout=predicator_dropout, n_out=n_out)
        preds = model(node_feats, adj, is_training)
        logits = clipped_sigmoid(preds)
        return logits

    model = hk.transform(forward, apply_rng=True)
    optimizer = optix.adam(learning_rate=lr)

    # define training loss
    @jax.jit
    def loss(params: hk.Params, key: PRNGKey, batch: Batch) -> jnp.ndarray:
        """Compute the loss (binary cross entropy) """
        inputs, targets = batch[:-1], batch[-1]
        logits = model.apply(params, key, *inputs)
        loss = -jnp.mean(targets * jnp.log(logits) + (1 - targets) * jnp.log(1 - logits))
        return loss

    # define training update
    @jax.jit
    def update(params: hk.Params, rng_key: PRNGKey,
               opt_state: OptState, batch: Batch) -> Tuple[hk.Params, OptState]:
        """Update the params"""
        grads = jax.grad(loss)(params, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optix.apply_updates(params, updates)
        return new_params, new_opt_state

    print("Starting training...")
    task_index = tox21_tasks.index(task)
    early_stop = EarlyStopping(patience=early_stop_patience)
    batch_init_data = (
        np.zeros((batch_size, *train_dataset.X[0][1].shape)),
        np.zeros((batch_size, *train_dataset.X[0][0].shape)),
        True
    )
    params = model.init(next(rng_seq), *batch_init_data)
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        # train
        start_time = time.time()
        for original_batch in train_dataset.iterbatches(batch_size=batch_size):
            batch = collate_fn(original_batch, task_index, True)
            params, opt_state = update(params, next(rng_seq), opt_state, batch)
        epoch_time = time.time() - start_time

        # valid
        y_score, y_true, valid_loss = [], [], []
        for original_batch in valid_dataset.iterbatches(batch_size=batch_size):
            batch = collate_fn(original_batch, task_index, False)
            y_score.extend(model.apply(params, next(rng_seq), *batch[:-1]))
            y_true.extend(batch[-1])
            valid_loss.append(loss(params, next(rng_seq), batch))
        score = roc_auc_score(y_true, y_score)

        # log
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) valid loss: {np.mean(valid_loss):.4f} \
            valid roc_auc score: {score:.4f}")
        # check early stopping
        early_stop.update(score, params)
        if early_stop.is_train_stop:
            print("Early stopping...")
            break

    # test
    y_score, y_true = [], []
    best_params = early_stop.best_params
    for original_batch in test_dataset.iterbatches(batch_size=batch_size):
        batch = collate_fn(original_batch, task_index, False)
        y_score.extend(model.apply(params, next(rng_seq), *batch[:-1]))
        y_true.extend(batch[-1])
    score = roc_auc_score(y_true, y_score)
    print(f'Test roc_auc score: {score:.4f}')
    # save best params
    with open('./best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)


if __name__ == "__main__":
    main()
