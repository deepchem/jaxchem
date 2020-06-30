import time
import math
import pickle
import argparse
from typing import Any, Tuple

import jax
import haiku as hk
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import optix
from sklearn.metrics import roc_auc_score


from deepchem.molnet import load_tox21
from jaxchem.models import PadGCNPredicator as GCNPredicator, clipped_sigmoid
from jaxchem.utils import EarlyStopping
from tox21_utils import disk_dataset_to_pad_pattern_data, shuffle_pad_pattern_data, create_batch


# type definition
State, OptState = Any, Any
ParamsAndStates = Tuple[hk.Params, State, OptState]
PadPatternData = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
MetricsData = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

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


def binary_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute binary cross entropy loss."""
    return -jnp.mean(targets * jnp.log(logits) + (1.0 - targets) * jnp.log(1.0 - logits))


def main():
    args = parse_arguments()

    # define hyperparams
    rng_seq = hk.PRNGSequence(args.seed)
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    task = args.task
    early_stop_patience = args.early_stop

    # setup data
    tox21_tasks, tox21_datasets, _ = load_tox21(featurizer='AdjacencyConv', reload=True)
    train_dataset, valid_dataset, test_dataset = tox21_datasets
    task_index = tox21_tasks.index(task)
    train_data = disk_dataset_to_pad_pattern_data(train_dataset, task_index)
    valid_data = disk_dataset_to_pad_pattern_data(valid_dataset, task_index)
    test_data = disk_dataset_to_pad_pattern_data(valid_dataset, task_index)

    # setup model and optimizer
    in_feats = train_dataset.X[0][1].shape[1]
    hidden_feats = [64, 64, 64]
    activation, batch_norm, dropout = None, None, None  # use default
    predicator_hidden_feats = 32
    pooling_method = 'mean'
    predicator_dropout = None  # use default
    n_out = 1  # binary classification

    def forward(node_feats: jnp.ndarray, adj: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        """Forward application of the GCN."""
        model = GCNPredicator(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation,
                              batch_norm=batch_norm, dropout=dropout, pooling_method=pooling_method,
                              predicator_hidden_feats=predicator_hidden_feats,
                              predicator_dropout=predicator_dropout, n_out=n_out)
        preds = model(node_feats, adj, is_training)
        logits = clipped_sigmoid(preds)
        return logits

    model = hk.transform_with_state(forward)
    optimizer = optix.adam(learning_rate=lr)

    # define loss function
    def loss(params: hk.Params, state: State, batch: PadPatternData) -> Tuple[jnp.ndarray, State]:
        """Compute the loss."""
        node_feats, adj, targets = batch
        logits, new_state = model.apply(params, state, next(rng_seq), node_feats, adj, True)
        loss = binary_cross_entropy(logits, targets)
        return loss, new_state

    @jax.jit
    def run_train(params_and_states: ParamsAndStates,
                  data: PadPatternData) -> Tuple[ParamsAndStates, PadPatternData]:
        """Update params and states for each epoch."""
        num_batch = math.ceil(len(data[0]) / batch_size)
        data = shuffle_pad_pattern_data(next(rng_seq), data)

        def loop_fun(i: int, params_and_states: ParamsAndStates) -> ParamsAndStates:
            """Update function for params and states about each batch data."""
            batch = create_batch(i, batch_size, data)
            params, state, opt_state = params_and_states
            (_, new_state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optix.apply_updates(params, updates)
            return (new_params, new_state, new_opt_state)

        return lax.fori_loop(0, num_batch, loop_fun, params_and_states), data

    @jax.jit
    def run_evaluate(params_and_states: ParamsAndStates,
                     data: PadPatternData) -> Tuple[ParamsAndStates, MetricsData]:
        """Compute evaluate metrics."""

        def loop_fun(carry: ParamsAndStates, data: MetricsData) -> Tuple[ParamsAndStates, MetricsData]:
            """Update function for params and states about each batch data."""
            node_feats, adj, targets = data
            node_feats, adj = jnp.expand_dims(node_feats, 0), jnp.expand_dims(adj, 0)
            params, state, _ = carry
            logits, _ = model.apply(params, state, None, node_feats, adj, False)
            loss = binary_cross_entropy(logits, targets)
            return carry, (logits, targets, loss)

        return lax.scan(loop_fun, params_and_states, data)

    print("Starting training...")
    early_stop = EarlyStopping(patience=early_stop_patience)
    batch_init_data = (
        jnp.zeros((batch_size, *train_dataset.X[0][1].shape)),
        jnp.zeros((batch_size, *train_dataset.X[0][0].shape)),
        True
    )
    params, state = model.init(next(rng_seq), *batch_init_data)
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        # train
        start_time = time.time()
        (params, state, opt_state), train_data = run_train((params, state, opt_state), train_data)
        epoch_time = time.time() - start_time
        # validation
        _, (y_valid_logits, y_valid_true, valid_loss) = run_evaluate((params, state, opt_state), valid_data)
        y_valid_logits = y_valid_logits.reshape(-1, 1)
        score = roc_auc_score(y_valid_true, y_valid_logits)
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) valid loss: {jnp.mean(valid_loss):.4f} \
            valid roc_auc score: {score:.4f}")
        # check early stopping
        early_stop.update(score, (params, state, opt_state))
        if early_stop.is_train_stop:
            print("Early stopping...")
            break

    # test
    best_checkpoints = early_stop.best_checkpoints
    _, (y_test_logits, y_test_true, _) = run_evaluate(best_checkpoints, test_data)
    y_test_logits = y_test_logits.reshape(-1, 1)
    score = roc_auc_score(y_test_true, y_test_logits)
    print(f'Test roc_auc score: {score:.4f}')
    # save best checkpoints
    with open('./best_checkpoints.pkl', 'wb') as f:
        pickle.dump(best_checkpoints, f)


if __name__ == "__main__":
    main()
