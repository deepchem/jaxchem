import os
import time
import random
import pickle
import argparse
from typing import Any, Tuple
from functools import partial

import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix
from sklearn.metrics import roc_auc_score


from deepchem.molnet import load_tox21
from jaxchem.models import SparseGCNPredicator as GCNPredicator, clipped_sigmoid
from jaxchem.utils import EarlyStopping


# type definition
Batch = Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
State, OptState = Any, Any

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


def collate_fn(original_batch: Any, task_index: int) -> Batch:
    """Make batch data as SparseGCN model inputs."""
    inputs, targets, _, idx = original_batch
    batch_size = len(inputs)
    node_feats = np.concatenate([inputs[i].atom_features for i in range(batch_size)], axis=0)
    src_idx, dest_idx, graph_idx = [], [], []
    total_n_atom = 0
    for i in range(batch_size):
        adj_list = inputs[i].canon_adj_list
        for j, edge in enumerate(adj_list):
            src_idx.extend([j + total_n_atom] * len(edge))
            dest_idx.extend(np.array(edge) + total_n_atom)
        graph_idx.extend([i] * inputs[i].n_atoms)
        total_n_atom += inputs[i].n_atoms
    edge_list = np.array([src_idx, dest_idx], dtype=np.int32)
    graph_idx = np.array(graph_idx, dtype=np.int32)
    targets = targets[:, task_index]
    return ((node_feats, edge_list), targets), graph_idx


def binary_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute binary cross entropy loss."""
    return -jnp.mean(targets * jnp.log(logits) + (1.0 - targets) * jnp.log(1.0 - logits))


def main():
    args = parse_arguments()
    # fix seed
    seed_everything(args.seed)

    # load tox21 dataset
    tox21_tasks, tox21_datasets, _ = load_tox21(featurizer='GraphConv', reload=True)
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    # define hyperparams
    rng_seq = hk.PRNGSequence(args.seed)
    # model params
    in_feats = train_dataset.X[0].n_feat
    hidden_feats = [64, 64, 64]
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
    def forward(node_feats: jnp.ndarray, adj: jnp.ndarray, graph_idx: jnp.ndarray,
                is_training: bool) -> jnp.ndarray:
        """Forward application of the GCN."""
        model = GCNPredicator(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation,
                              batch_norm=batch_norm, dropout=dropout, pooling_method=pooling_method,
                              predicator_hidden_feats=predicator_hidden_feats,
                              predicator_dropout=predicator_dropout, n_out=n_out)
        preds = model(node_feats, adj, graph_idx, is_training)
        logits = clipped_sigmoid(preds)
        return logits

    model = hk.transform_with_state(forward)
    optimizer = optix.adam(learning_rate=lr)

    # define training loss
    def train_loss(params: hk.Params, state: State, batch: Batch,
                   graph_idx: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
        """Compute the loss."""
        inputs, targets = batch
        logits, new_state = model.apply(params, state, next(rng_seq), *inputs, graph_idx, True)
        loss = binary_cross_entropy(logits, targets)
        return loss, new_state

    # define training update
    @partial(jax.jit, static_argnums=(4,))
    def update(params: hk.Params, state: State, opt_state: OptState,
               batch: Batch, graph_idx: jnp.ndarray) -> Tuple[hk.Params, State, OptState]:
        """Update the params."""
        (_, new_state), grads = jax.value_and_grad(train_loss, has_aux=True)(params, state, batch, graph_idx)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optix.apply_updates(params, updates)
        return new_params, new_state, new_opt_state

    # define evaluate metrics
    @partial(jax.jit, static_argnums=(3,))
    def evaluate(params: hk.Params, state: State, batch: Batch,
                 graph_idx: jnp.ndarray) -> jnp.ndarray:
        """Compute evaluate metrics."""
        inputs, targets = batch
        logits, _ = model.apply(params, state, next(rng_seq), *inputs, graph_idx, False)
        loss = binary_cross_entropy(logits, targets)
        return logits, loss, targets

    print("Starting training...")
    task_index = tox21_tasks.index(task)
    early_stop = EarlyStopping(patience=early_stop_patience)
    init_batch, init_graph_idx = collate_fn(
        next(train_dataset.iterbatches(batch_size=batch_size)), task_index
    )
    params, state = model.init(next(rng_seq), *init_batch[0], init_graph_idx, True)
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        # train
        start_time = time.time()
        # FIXME : This for loop should be rewrited by lax.scan or lax.fori_loop
        # update operation is tii slow....
        for original_batch in train_dataset.iterbatches(batch_size=batch_size):
            batch, graph_idx = collate_fn(original_batch, task_index)
            params, state, opt_state = update(params, state, opt_state, batch, graph_idx)
        epoch_time = time.time() - start_time

        # valid
        y_score, y_true, valid_loss = [], [], []
        for original_batch in valid_dataset.iterbatches(batch_size=batch_size):
            batch, graph_idx = collate_fn(original_batch, task_index)
            logits, loss, targets = evaluate(params, state, batch, graph_idx)
            y_score.extend(logits), valid_loss.append(loss), y_true.extend(targets)
        score = roc_auc_score(y_true, y_score)

        # log
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) valid loss: {np.mean(valid_loss):.4f} \
            valid roc_auc score: {score:.4f}")
        # check early stopping
        early_stop.update(score, (params, state))
        if early_stop.is_train_stop:
            print("Early stopping...")
            break

    # test
    y_score, y_true = [], []
    best_checkpoints = early_stop.best_checkpoints
    for original_batch in test_dataset.iterbatches(batch_size=batch_size):
        batch, graph_idx = collate_fn(original_batch, task_index)
        logits, _, targets = evaluate(*best_checkpoints, batch, graph_idx)
        y_score.extend(logits), y_true.extend(targets)
    score = roc_auc_score(y_true, y_score)
    print(f'Test roc_auc score: {score:.4f}')
    # save best checkpoints
    with open('./best_checkpoints.pkl', 'wb') as f:
        pickle.dump(best_checkpoints, f)


if __name__ == "__main__":
    main()
