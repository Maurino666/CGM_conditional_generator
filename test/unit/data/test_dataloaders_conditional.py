import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import create_conditional_dataloaders


def test_create_conditional_dataloaders_batch_shapes_and_types():
    """
    DataLoaders must yield batches of (y, c) with expected shapes and float32 dtype.
    """
    n_train = 10
    n_val = 4
    seq_len = 7
    target_dim = 1
    cond_dim = 3

    y_train = np.random.rand(n_train, seq_len, target_dim).astype(np.float32)
    c_train = np.random.rand(n_train, seq_len, cond_dim).astype(np.float32)
    y_val = np.random.rand(n_val, seq_len, target_dim).astype(np.float32)
    c_val = np.random.rand(n_val, seq_len, cond_dim).astype(np.float32)

    batch_size = 4

    train_loader, val_loader = create_conditional_dataloaders(
        y_train=y_train,
        c_train=c_train,
        y_val=y_val,
        c_val=c_val,
        batch_size=batch_size,
        shuffle_train=True,
        num_workers=0,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Check dataset lengths
    assert len(train_loader.dataset) == n_train
    assert len(val_loader.dataset) == n_val

    # Inspect first training batch
    train_batch = next(iter(train_loader))
    y_batch, c_batch = train_batch

    # Expected batch size may be smaller for the last batch, but for the first
    # we expect full batch_size since n_train >= batch_size
    assert y_batch.shape == (batch_size, seq_len, target_dim)
    assert c_batch.shape == (batch_size, seq_len, cond_dim)

    assert y_batch.dtype == torch.float32
    assert c_batch.dtype == torch.float32

    # Gather all samples from the train loader to ensure no samples are lost
    total_y = 0
    for y_b, _ in train_loader:
        total_y += y_b.shape[0]
    assert total_y == n_train


def test_create_conditional_dataloaders_mismatched_shapes_raise():
    """
    If targets and conditionals have mismatched shapes along num_windows or seq_len,
    the underlying ConditionalTimeSeriesDataset must raise ValueError.
    """
    # Mismatch in num_windows
    y_train = np.zeros((5, 10, 1), dtype=np.float32)
    c_train = np.zeros((4, 10, 2), dtype=np.float32)  # different n_windows
    y_val = np.zeros((3, 10, 1), dtype=np.float32)
    c_val = np.zeros((3, 10, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = create_conditional_dataloaders(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            batch_size=2,
            shuffle_train=False,
            num_workers=0,
        )

    # Mismatch in seq_len
    y_train = np.zeros((5, 10, 1), dtype=np.float32)
    c_train = np.zeros((5, 8, 2), dtype=np.float32)  # different seq_len
    y_val = np.zeros((3, 10, 1), dtype=np.float32)
    c_val = np.zeros((3, 10, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = create_conditional_dataloaders(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            batch_size=2,
            shuffle_train=False,
            num_workers=0,
        )
