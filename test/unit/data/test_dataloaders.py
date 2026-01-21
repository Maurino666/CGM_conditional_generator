import numpy as np
import torch
from torch.utils.data import DataLoader
import pytest

from data import create_dataloaders, TimeSeriesDataset


def _make_dummy_windows(n_windows: int, seq_len: int, n_features: int) -> np.ndarray:
    return np.random.randn(n_windows, seq_len, n_features).astype(np.float32)


def test_create_dataloaders_basic():
    """Verifica che i DataLoader vengano creati correttamente e rispettino shape e batch_size."""
    X_train = _make_dummy_windows(10, 5, 3)
    X_val = _make_dummy_windows(4, 5, 3)

    batch_size = 4

    train_loader, val_loader = create_dataloaders(
        X_train=X_train,
        X_val=X_val,
        batch_size=batch_size,
        shuffle_train=False,  # disabilito shuffle per controllare l'ordine
        num_workers=0,
    )

    # Sono DataLoader
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Dataset interni sono TimeSeriesDataset
    assert isinstance(train_loader.dataset, TimeSeriesDataset)
    assert isinstance(val_loader.dataset, TimeSeriesDataset)

    # Lunghezze dataset
    assert len(train_loader.dataset) == X_train.shape[0]
    assert len(val_loader.dataset) == X_val.shape[0]

    # Controllo il contenuto del primo batch del train_loader
    train_batches = list(train_loader)
    first_batch = train_batches[0]
    assert isinstance(first_batch, torch.Tensor)
    # shape: (batch_size, seq_len, n_features) oppure ultimo batch più corto
    assert first_batch.shape[1:] == (5, 3)

    # Con shuffle_train=False, il primo batch deve corrispondere ai primi esempi di X_train
    expected = torch.from_numpy(X_train[: batch_size])
    torch.testing.assert_close(first_batch, expected)

    # Controllo val_loader (niente shuffle)
    val_batches = list(val_loader)
    val_first = val_batches[0]
    assert val_first.shape[1:] == (5, 3)
    expected_val = torch.from_numpy(X_val[: batch_size])
    torch.testing.assert_close(val_first, expected_val)


def test_create_dataloaders_invalid_dims_train():
    """Se X_train non è 3D, deve sollevare ValueError."""
    X_train = np.random.randn(10, 5)   # 2D
    X_val = _make_dummy_windows(4, 5, 3)

    with pytest.raises(ValueError, match="X_train must be 3D"):
        create_dataloaders(
            X_train=X_train,
            X_val=X_val,
            batch_size=4,
        )


def test_create_dataloaders_invalid_dims_val():
    """Se X_val non è 3D, deve sollevare ValueError."""
    X_train = _make_dummy_windows(10, 5, 3)
    X_val = np.random.randn(4, 5)  # 2D

    with pytest.raises(ValueError, match="X_val must be 3D"):
        create_dataloaders(
            X_train=X_train,
            X_val=X_val,
            batch_size=4,
        )
