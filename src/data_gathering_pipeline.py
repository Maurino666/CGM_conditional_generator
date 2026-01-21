from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from src.data import (
    create_dataloaders,
    minmax_scale_features,
    create_conditional_dataloaders,
    minmax_scale_conditional
)
from src.data_prep import AZT1D2025Dataset, HUPA_UCMDataset, OhioT1DMDataset


def gather_data(
        features: list[str],
        seq_len: int,
        step: int,
        val_ratio: float = 0.2,
        random_state: int | None = None,
        batch_size: int = 64,
        normalize: list[str] | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load all datasets, build sliding-window sequences, optionally normalize
    a subset of features, and create train/val DataLoaders.
    """

    # 1. Instantiate datasets and run cleaning pipeline
    dataset1 = AZT1D2025Dataset(
        Path("../datasets/AZT1D2025/CGM Records"),
        Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    dataset1.clean_data()

    dataset2 = HUPA_UCMDataset(
        Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )
    dataset2.clean_data()

    dataset3 = OhioT1DMDataset(
        Path("../datasets/OhioT1DMmini"),
        Path("../datasets/OhioT1DMmini/ohiot1dmmini.yaml"),
        logging_dir=Path("../datasets/OhioT1DMmini/prep_logs"),
    )
    dataset3.clean_data()

    print("Gathered dataset classes")

    # 2. Build sliding-window sequences for each dataset
    X_train1, X_val1 = dataset1.to_sequence_splits(
        seq_len=seq_len,
        step=step,
        feature_cols=features,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print("Train 1 windows: ", X_train1.shape)
    print("Val 1 windows:   ", X_val1.shape)

    X_train2, X_val2 = dataset2.to_sequence_splits(
        seq_len=seq_len,
        step=step,
        feature_cols=features,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print("Train 2 windows: ", X_train2.shape)
    print("Val 2 windows:   ", X_val2.shape)

    X_train3, X_val3 = dataset3.to_sequence_splits(
        seq_len=seq_len,
        step=step,
        feature_cols=features,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print("Train 3 windows: ", X_train3.shape)
    print("Val 3 windows:   ", X_val3.shape)

    # Check consistency of (seq_len, num_features)
    assert X_train1.shape[1:] == X_train2.shape[1:] == X_train3.shape[1:], \
        "Train arrays must have the same (seq_len, num_features)"
    assert X_val1.shape[1:] == X_val2.shape[1:] == X_val3.shape[1:], \
        "Validation arrays must have the same (seq_len, num_features)"

    # 3. Concatenate all training and validation windows
    X_train = np.concatenate([X_train1, X_train2, X_train3], axis=0)
    X_val = np.concatenate([X_val1, X_val2, X_val3], axis=0)

    print("Combined train shape: ", X_train.shape)
    print("Combined val shape:   ", X_val.shape)

    # 4. Optional Min–Max normalization on selected features
    if normalize:
        X_train, X_val = minmax_scale_features(
            X_train=X_train,
            X_val=X_val,
            features=features,
            normalize=normalize,
        )

    # 5. Build DataLoaders from normalized (or raw) arrays
    train_loader, val_loader = create_dataloaders(
        X_train=X_train,
        X_val=X_val,
        batch_size=batch_size,
        num_workers=4,
    )

    return train_loader, val_loader

def gather_data_conditional(
    cond_features: List[str],
    seq_len: int,
    step: int,
    val_ratio: float = 0.2,
    random_state: int | None = None,
    batch_size: int = 64,
    normalize: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load all datasets, build conditional sliding-window sequences, optionally
    normalize target + conditional features, and create train/val DataLoaders.

    The target column name is taken from each dataset (and must be consistent
    across them, e.g. 'glucose') while conditional columns are provided by
    `cond_features`.
    """

    # 1. Instantiate datasets and run cleaning pipeline
    dataset1 = AZT1D2025Dataset(
        Path("../datasets/AZT1D2025/CGM Records"),
        Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    dataset1.clean_data()

    dataset2 = HUPA_UCMDataset(
        Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )
    dataset2.clean_data()

    dataset3 = OhioT1DMDataset(
        Path("../datasets/OhioT1DMmini"),
        Path("../datasets/OhioT1DMmini/ohiot1dmmini.yaml"),
        logging_dir=Path("../datasets/OhioT1DMmini/prep_logs"),
    )
    dataset3.clean_data()

    print("Gathered dataset classes")

    # 2. Sanity check: target column name consistent across datasets
    target_col = dataset1.target_col
    assert dataset2.target_col == target_col == dataset3.target_col, (
        "All datasets must share the same target_col in the unified schema."
    )

    # 3. Build conditional sliding-window sequences for each dataset
    #    Each call returns: (y_train, c_train, y_val, c_val)
    y_train1, c_train1, y_val1, c_val1 = dataset1.to_sequence_splits_conditional(
        seq_len=seq_len,
        step=step,
        cond_cols=cond_features,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    print("Dataset 1 - train windows:", y_train1.shape, c_train1.shape)
    print("Dataset 1 - val   windows:", y_val1.shape, c_val1.shape)

    y_train2, c_train2, y_val2, c_val2 = dataset2.to_sequence_splits_conditional(
        seq_len=seq_len,
        step=step,
        cond_cols=cond_features,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    print("Dataset 2 - train windows:", y_train2.shape, c_train2.shape)
    print("Dataset 2 - val   windows:", y_val2.shape, c_val2.shape)

    y_train3, c_train3, y_val3, c_val3 = dataset3.to_sequence_splits_conditional(
        seq_len=seq_len,
        step=step,
        cond_cols=cond_features,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    print("Dataset 3 - train windows:", y_train3.shape, c_train3.shape)
    print("Dataset 3 - val   windows:", y_val3.shape, c_val3.shape)

    # 4. Check consistency of shapes (seq_len, target_dim/cond_dim)
    assert y_train1.shape[1:] == y_train2.shape[1:] == y_train3.shape[1:], (
        "Train target arrays must have the same (seq_len, target_dim)."
    )
    assert c_train1.shape[1:] == c_train2.shape[1:] == c_train3.shape[1:], (
        "Train conditional arrays must have the same (seq_len, cond_dim)."
    )
    assert y_val1.shape[1:] == y_val2.shape[1:] == y_val3.shape[1:], (
        "Val target arrays must have the same (seq_len, target_dim)."
    )
    assert c_val1.shape[1:] == c_val2.shape[1:] == c_val3.shape[1:], (
        "Val conditional arrays must have the same (seq_len, cond_dim)."
    )

    # 5. Concatenate all training and validation windows
    y_train = np.concatenate([y_train1, y_train2, y_train3], axis=0)
    c_train = np.concatenate([c_train1, c_train2, c_train3], axis=0)
    y_val = np.concatenate([y_val1, y_val2, y_val3], axis=0)
    c_val = np.concatenate([c_val1, c_val2, c_val3], axis=0)

    print("Combined train target shape: ", y_train.shape)
    print("Combined train cond   shape: ", c_train.shape)
    print("Combined val   target shape: ", y_val.shape)
    print("Combined val   cond   shape: ", c_val.shape)

    # 6. Optional Min–Max normalization on selected features
    if normalize:
        # Normalize target + cond jointly, then split
        y_train, c_train, y_val, c_val = minmax_scale_conditional(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            target_feature=target_col,
            cond_features=cond_features,
            normalize=normalize,
        )

    # 7. Build conditional DataLoaders
    train_loader, val_loader = create_conditional_dataloaders(
        y_train=y_train,
        c_train=c_train,
        y_val=y_val,
        c_val=c_val,
        batch_size=batch_size,
        shuffle_train=True,
        num_workers=4,
    )

    return train_loader, val_loader
