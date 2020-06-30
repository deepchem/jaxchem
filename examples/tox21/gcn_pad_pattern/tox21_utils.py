from typing import Any, Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom


PRNGKey = jnp.ndarray
PadPatternData = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


# there is a possibility for DiskDataset to be able to include this function like `make_pytorch_dataset`
def disk_dataset_to_pad_pattern_data(disk_dataset: Any, task_index: int) -> PadPatternData:
    """Convert DiskDataset to jax.numpy array"""
    inputs, targets = disk_dataset.X, disk_dataset.y
    node_feats = jnp.array([inputs[i][1] for i in range(len(inputs))])
    adj = jnp.array([inputs[i][0] for i in range(len(inputs))])
    targets = jnp.array(targets[:, task_index])
    return node_feats, adj, targets


def shuffle_pad_pattern_data(key: PRNGKey, data: PadPatternData) -> PadPatternData:
    train_node_feats, train_adj, train_y = data
    shuffle_ids = jrandom.permutation(key, jnp.arange(len(train_node_feats)))
    data = train_node_feats[shuffle_ids], train_adj[shuffle_ids], train_y[shuffle_ids]
    return data


def create_batch(ith_batch: int, batch_size: int, data: PadPatternData) -> PadPatternData:
    batch = [None for _ in range(len(data))]
    for idx in range(len(data)):
        batch[idx] = lax.dynamic_slice_in_dim(data[idx], ith_batch * batch_size, batch_size)
    return tuple(batch)


