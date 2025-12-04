import numpy as np
import pytest

from data import minmax_scale_features


def _make_simple_arrays():
    """
    Costruisce piccoli X_train/X_val con 3 feature:
      f0: non normalizzata
      f1: range ampio
      f2: range più piccolo
    """
    # shape: (n_windows, seq_len, n_features)
    X_train = np.array(
        [
            # window 0
            [
                [1.0, 10.0, 0.0],
                [2.0, 20.0, 1.0],
            ],
            # window 1
            [
                [3.0, 30.0, 2.0],
                [4.0, 40.0, 3.0],
            ],
        ],
        dtype=float,
    )
    X_val = np.array(
        [
            [
                [5.0, 15.0, -1.0],
                [6.0, 35.0, 4.0],
            ]
        ],
        dtype=float,
    )
    features = ["f0", "f1", "f2"]
    return X_train, X_val, features


def test_minmax_scale_features_basic():
    """Controlla che la normalizzazione Min–Max sia corretta sulle feature selezionate."""
    X_train, X_val, features = _make_simple_arrays()
    normalize = ["f1", "f2"]

    X_train_scaled, X_val_scaled = minmax_scale_features(
        X_train, X_val, features, normalize
    )

    # La shape deve restare invariata
    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_val.shape

    # La feature non normalizzata (f0) deve restare identica
    np.testing.assert_allclose(X_train_scaled[:, :, 0], X_train[:, :, 0])
    np.testing.assert_allclose(X_val_scaled[:, :, 0], X_val[:, :, 0])

    # Sui dati di train:
    # per f1 (indice 1), min=10, max=40 → valori in [0,1]
    f1_train = X_train_scaled[:, :, 1]
    assert np.isclose(f1_train.min(), 0.0, atol=1e-6)
    assert np.isclose(f1_train.max(), 1.0, atol=1e-6)

    # per f2 (indice 2), controlliamo anche qui [0,1]
    f2_train = X_train_scaled[:, :, 2]
    assert np.isclose(f2_train.min(), 0.0, atol=1e-6)
    assert np.isclose(f2_train.max(), 1.0, atol=1e-6)

    # Gli originali non devono essere modificati in-place
    # (giusto un controllo leggero: se i min/max sono ancora quelli originali)
    assert X_train[:, :, 1].min() == 10.0
    assert X_train[:, :, 1].max() == 40.0


def test_minmax_scale_features_constant_feature_sets_to_zero():
    """Se una feature è quasi costante nel train, deve diventare tutta zero in train e val."""
    # X_train: feature f1 sempre 5.0
    X_train = np.array(
        [
            [[1.0, 5.0], [2.0, 5.0]],
            [[3.0, 5.0], [4.0, 5.0]],
        ],
        dtype=float,
    )
    # X_val con altri valori (ma verranno ignorati)
    X_val = np.array(
        [
            [[10.0, 7.0], [11.0, 9.0]],
        ],
        dtype=float,
    )
    features = ["f0", "f1"]
    normalize = ["f1"]

    X_train_scaled, X_val_scaled = minmax_scale_features(
        X_train, X_val, features, normalize
    )

    # f1 (indice 1) deve essere tutta zero
    assert np.allclose(X_train_scaled[:, :, 1], 0.0)
    assert np.allclose(X_val_scaled[:, :, 1], 0.0)

    # f0 invariata
    np.testing.assert_allclose(X_train_scaled[:, :, 0], X_train[:, :, 0])
    np.testing.assert_allclose(X_val_scaled[:, :, 0], X_val[:, :, 0])


def test_minmax_scale_features_missing_feature_raises():
    """Se normalize contiene colonne non presenti in features, deve sollevare ValueError."""
    X_train = np.zeros((2, 3, 2), dtype=float)
    X_val = np.zeros((1, 3, 2), dtype=float)
    features = ["a", "b"]
    normalize = ["b", "c"]  # 'c' non esiste

    with pytest.raises(ValueError, match="normalize columns"):
        minmax_scale_features(X_train, X_val, features, normalize)


def test_minmax_scale_features_noop_when_normalize_empty():
    """Se normalize è vuoto, la funzione deve restituire i dati invariati."""
    X_train = np.random.randn(2, 3, 2)
    X_val = np.random.randn(1, 3, 2)
    features = ["a", "b"]
    normalize: list[str] = []

    X_train_scaled, X_val_scaled = minmax_scale_features(
        X_train, X_val, features, normalize
    )

    np.testing.assert_allclose(X_train_scaled, X_train)
    np.testing.assert_allclose(X_val_scaled, X_val)
