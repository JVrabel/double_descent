"""
todo
"""
import os
import numpy as onp
import jax
import jax.numpy as jnp


def create_dataloaders(
    train_dir: str, 
    train_labels_dir: str, 
    test_dir: str, 
    test_labels_dir: str, 
    batch_size: int, 
    ):
    """Creates training and testing DataLoaders.


    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.

    batch_size: Number of samples per batch in each of the DataLoaders.


    Returns:

    """
    # Use ImageFolder to create dataset(s)
    train_data = onp.load(train_dir)
    test_data = onp.load(test_dir)

    # Get class names
    train_labels = onp.load(train_labels_dir)
    test_labels = onp.load(test_labels_dir)

    train_labels = jnp.squeeze(jax.nn.one_hot((train_labels), num_classes=10))
    test_labels = jnp.squeeze(jax.nn.one_hot((test_labels), num_classes=10))


    # Turn data into data loaders
    num_train = train_data.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def train_data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]
        return train_data_stream    
    train_dataloader = train_data_stream()

    def valid_data_stream(): #todo batch?
        while True:
            yield test_data, test_labels
        return valid_data_stream
    test_dataloader = valid_data_stream

    return train_dataloader, test_dataloader, train_labels
