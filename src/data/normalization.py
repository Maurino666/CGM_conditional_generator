import numpy as np

def minmax_scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    features: list[str],
    normalize: list[str],
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply feature-wise Min–Max scaling to a subset of features.

    Scaling is computed ONLY on the training data and then applied
    to both train and validation arrays.

    Parameters
    ----------
    X_train : np.ndarray
        Training windows, shape (n_train_windows, seq_len, n_features).
    X_val : np.ndarray
        Validation windows, shape (n_val_windows, seq_len, n_features).
    features : list[str]
        Ordered list of feature names corresponding to the last dimension
        of X_train / X_val.
    normalize : list[str]
        Subset of `features` to be Min–Max scaled to [0, 1].
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    X_train_scaled, X_val_scaled : np.ndarray, np.ndarray
        Normalized copies of the input arrays.
    """
    if not normalize:
        # Nothing to do
        return X_train, X_val

    normalize_set = set(normalize)
    feature_set = set(features)

    # Ensure all requested columns are present in features
    if not normalize_set.issubset(feature_set):
        missing = normalize_set - feature_set
        raise ValueError(
            f"normalize columns {sorted(missing)} are not present in features."
        )

    # Work on copies to avoid mutating the original arrays in-place
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    # Map feature names to indices in the last dimension
    normalize_indices = [features.index(col) for col in normalize]

    for col_name, idx in zip(normalize, normalize_indices):
        # Take all values of this feature from training windows
        col_train = X_train_scaled[:, :, idx]  # shape (n_train_windows, seq_len)
        col_min = float(col_train.min())
        col_max = float(col_train.max())

        if abs(col_max - col_min) < eps:
            # Feature is (almost) constant on train: normalization is not meaningful.
            # Set it to 0.0 in both train and val.
            print(
                f"[minmax_scale_features] Warning: feature '{col_name}' is "
                "quasi-constant on training data; setting normalized values to 0."
            )
            X_train_scaled[:, :, idx] = 0.0
            X_val_scaled[:, :, idx] = 0.0
            continue

        scale = col_max - col_min

        # Apply Min–Max scaling to train and val for this feature
        X_train_scaled[:, :, idx] = (col_train - col_min) / (scale + eps)
        X_val_scaled[:, :, idx] = (
            X_val_scaled[:, :, idx] - col_min
        ) / (scale + eps)

    return X_train_scaled, X_val_scaled
