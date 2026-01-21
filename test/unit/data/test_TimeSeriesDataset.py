import numpy as np
import torch
import pytest

from data import TimeSeriesDataset


def test_timeseriesdataset_init_3d_ok():
    """Costruzione corretta con array 3D."""
    num_windows = 5
    seq_len = 10
    num_features = 3

    arr = np.random.randn(num_windows, seq_len, num_features)
    ds = TimeSeriesDataset(arr)

    # Lunghezza corretta
    assert len(ds) == num_windows

    # I dati sono float32
    assert ds.sequences.dtype == np.float32

    # __getitem__ restituisce un Tensor con la shape giusta
    x0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape == (seq_len, num_features)
    assert x0.dtype == torch.float32


def test_timeseriesdataset_init_raises_on_non_3d():
    """Se l'array non è 3D, deve sollevare ValueError."""
    arr_2d = np.random.randn(10, 3)
    with pytest.raises(ValueError, match="must be a 3D array"):
        TimeSeriesDataset(arr_2d)

    arr_4d = np.random.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="must be a 3D array"):
        TimeSeriesDataset(arr_4d)


def test_timeseriesdataset_indexing_consistency():
    """Controlla che l'i-esimo elemento corrisponda ai dati originali."""
    arr = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    ds = TimeSeriesDataset(arr)

    for i in range(len(ds)):
        x = ds[i]
        np.testing.assert_allclose(x.numpy(), arr[i].astype(np.float32))
