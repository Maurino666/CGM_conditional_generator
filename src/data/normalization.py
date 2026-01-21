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


def minmax_scale_conditional(
    y_train: np.ndarray,
    c_train: np.ndarray,
    y_val: np.ndarray,
    c_val: np.ndarray,
    target_feature: str,
    cond_features: list[str],
    normalize: list[str],
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Min–Max scaling to target + conditional features using the existing
    minmax_scale_features helper.

    Parameters
    ----------
    y_train, y_val : np.ndarray
        Target sequences, shape (n_windows, seq_len, 1).
    c_train, c_val : np.ndarray
        Conditional sequences, shape (n_windows, seq_len, cond_dim).
    target_feature : str
        Name of the target feature (e.g. 'glucose').
    cond_features : list[str]
        Names of conditional features, in the same order as c_train/c_val last dim.
    normalize : list[str]
        Subset of [target_feature] + cond_features to scale.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    y_train_scaled, c_train_scaled, y_val_scaled, c_val_scaled
    """
    if not normalize:
        return y_train, c_train, y_val, c_val

    # 1) Build full feature arrays by concatenating target and conditionals
    X_train_full = np.concatenate([y_train, c_train], axis=-1)
    X_val_full = np.concatenate([y_val, c_val], axis=-1)

    # 2) Build the feature list consistent with concatenation order
    all_features = [target_feature] + cond_features

    # 3) Reuse existing scaler
    from src.data import minmax_scale_features  # or appropriate import

    X_train_scaled, X_val_scaled = minmax_scale_features(
        X_train=X_train_full,
        X_val=X_val_full,
        features=all_features,
        normalize=normalize,
        eps=eps,
    )

    # 4) Split back into target and conditionals
    y_train_scaled = X_train_scaled[:, :, :1]
    c_train_scaled = X_train_scaled[:, :, 1:]
    y_val_scaled = X_val_scaled[:, :, :1]
    c_val_scaled = X_val_scaled[:, :, 1:]

    return y_train_scaled, c_train_scaled, y_val_scaled, c_val_scaled

