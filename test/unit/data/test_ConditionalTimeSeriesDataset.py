import numpy as np
import pytest
import torch

from data import ConditionalTimeSeriesDataset


def test_conditional_timeseries_dataset_basic_len_and_getitem():
    """Dataset must expose correct length and return (y, c) with expected shapes and dtypes."""
    num_windows = 5
    seq_len = 7
    target_dim = 1
    cond_dim = 3

    targets = np.random.rand(num_windows, seq_len, target_dim).astype(np.float32)
    conditionals = np.random.rand(num_windows, seq_len, cond_dim).astype(np.float32)

    dataset = ConditionalTimeSeriesDataset(targets, conditionals)

    # __len__ must match num_windows
    assert len(dataset) == num_windows

    # __getitem__ must return two tensors with correct shapes and dtypes
    idx = 2
    y, c = dataset[idx]

    assert isinstance(y, torch.Tensor)
    assert isinstance(c, torch.Tensor)

    assert y.shape == (seq_len, target_dim)
    assert c.shape == (seq_len, cond_dim)

    assert y.dtype == torch.float32
    assert c.dtype == torch.float32

    # Values must match the original NumPy arrays at the same index
    np.testing.assert_allclose(y.numpy(), targets[idx], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c.numpy(), conditionals[idx], rtol=1e-6, atol=1e-6)


def test_conditional_timeseries_dataset_requires_3d_arrays():
    """targets and conditionals must be 3D arrays; otherwise a ValueError is raised."""
    # 2D targets
    targets_2d = np.zeros((10, 4), dtype=np.float32)
    conditionals_3d = np.zeros((10, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = ConditionalTimeSeriesDataset(targets_2d, conditionals_3d)

    # 2D conditionals
    targets_3d = np.zeros((10, 4, 1), dtype=np.float32)
    conditionals_2d = np.zeros((10, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = ConditionalTimeSeriesDataset(targets_3d, conditionals_2d)


def test_conditional_timeseries_dataset_mismatched_num_windows_raises():
    """If num_windows differ between targets and conditionals, a ValueError is raised."""
    targets = np.zeros((5, 4, 1), dtype=np.float32)
    conditionals = np.zeros((4, 4, 2), dtype=np.float32)  # different num_windows

    with pytest.raises(ValueError):
        _ = ConditionalTimeSeriesDataset(targets, conditionals)


def test_conditional_timeseries_dataset_mismatched_seq_len_raises():
    """If seq_len differ between targets and conditionals, a ValueError is raised."""
    targets = np.zeros((5, 4, 1), dtype=np.float32)
    conditionals = np.zeros((5, 3, 2), dtype=np.float32)  # different seq_len

    with pytest.raises(ValueError):
        _ = ConditionalTimeSeriesDataset(targets, conditionals)
