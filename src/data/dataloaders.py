from .TimeSeriesDataset import TimeSeriesDataset

import numpy as np
from torch.utils.data import DataLoader


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation from two arrays
    of time series windows.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences, shape (num_train_windows, seq_len, num_features).
    X_val : np.ndarray
        Validation sequences, shape (num_val_windows, seq_len, num_features).
    batch_size : int
        Batch size for both DataLoaders.
    shuffle_train : bool, optional
        Whether to shuffle the training dataset at each epoch.
    num_workers : int, optional
        Number of worker processes for the DataLoader (0 = no multiprocessing).

    Returns
    -------
    (train_loader, val_loader) : tuple of DataLoader
        DataLoaders for training and validation.
    """
    if X_train.ndim != 3:
        raise ValueError(
            f"X_train must be 3D (num_windows, seq_len, num_features), "
            f"got shape {X_train.shape}"
        )
    if X_val.ndim != 3:
        raise ValueError(
            f"X_val must be 3D (num_windows, seq_len, num_features), "
            f"got shape {X_val.shape}"
        )

    train_dataset = TimeSeriesDataset(X_train)
    val_dataset = TimeSeriesDataset(X_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader
