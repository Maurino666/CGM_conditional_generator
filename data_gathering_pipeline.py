from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from data import create_dataloaders, minmax_scale_features
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset, OhioT1DMDataset

def gather_data(
        features: List[str],
        seq_len: int,
        step: int,
        val_ratio: float = 0.2,
        random_state: int | None = None,
        batch_size: int = 64,
        normalize: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load all datasets, build sliding-window sequences, optionally normalize
    a subset of features, and create train/val DataLoaders.
    """

    # 1. Instantiate datasets and run cleaning pipeline
    dataset1 = AZT1D2025Dataset(
        Path("datasets/AZT1D2025/CGM Records"),
        Path("datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        logging_dir=Path("datasets/AZT1D2025/prep_logs"),
    )
    dataset1.clean_data()

    dataset2 = HUPA_UCMDataset(
        Path("datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        Path("datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        logging_dir=Path("datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )
    dataset2.clean_data()

    dataset3 = OhioT1DMDataset(
        Path("datasets/OhioT1DMmini"),
        Path("datasets/OhioT1DMmini/ohiot1dmmini.yaml"),
        logging_dir=Path("datasets/OhioT1DMmini/prep_logs"),
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
