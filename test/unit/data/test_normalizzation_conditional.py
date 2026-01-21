import numpy as np
import pytest

from data import minmax_scale_conditional


def test_minmax_scale_conditional_no_normalize_returns_inputs():
    """If normalize is empty, the function must return the original arrays unchanged."""
    y_train = np.random.rand(5, 4, 1)
    c_train = np.random.rand(5, 4, 2)
    y_val = np.random.rand(3, 4, 1)
    c_val = np.random.rand(3, 4, 2)

    y_scaled, c_scaled, y_val_scaled, c_val_scaled = minmax_scale_conditional(
        y_train=y_train,
        c_train=c_train,
        y_val=y_val,
        c_val=c_val,
        target_feature="glucose",
        cond_features=["bolus", "carbs"],
        normalize=[],  # nothing to scale
    )

    # Same objects and same values
    assert y_scaled is y_train
    assert c_scaled is c_train
    assert y_val_scaled is y_val
    assert c_val_scaled is c_val


def test_minmax_scale_conditional_scales_target_and_conditional():
    """
    Target and conditional features are min–max scaled independently using
    training statistics, and the same transform is applied to validation data.
    """
    # Shapes: (n_windows, seq_len, dim)
    # Target feature values on train: [0, 10, 5, 15] -> min=0, max=15
    y_train = np.array(
        [
            [[0.0], [10.0]],
            [[5.0], [15.0]],
        ],
        dtype=np.float32,
    )
    # Cond feature values on train: [0, 100, 50, 150] -> min=0, max=150
    c_train = np.array(
        [
            [[0.0], [100.0]],
            [[50.0], [150.0]],
        ],
        dtype=np.float32,
    )

    # Validation values outside the train range to check extrapolation
    # Target: 20 -> (20 - 0) / 15
    # Cond:   200 -> (200 - 0) / 150
    y_val = np.array([[[20.0], [0.0]]], dtype=np.float32)
    c_val = np.array([[[200.0], [0.0]]], dtype=np.float32)

    y_train_scaled, c_train_scaled, y_val_scaled, c_val_scaled = minmax_scale_conditional(
        y_train=y_train,
        c_train=c_train,
        y_val=y_val,
        c_val=c_val,
        target_feature="glucose",
        cond_features=["bolus"],
        normalize=["glucose", "bolus"],
    )

    # Check shapes preserved
    assert y_train_scaled.shape == y_train.shape
    assert c_train_scaled.shape == c_train.shape
    assert y_val_scaled.shape == y_val.shape
    assert c_val_scaled.shape == c_val.shape

    # Expected scaled train values
    # target_train_scaled = original / 15
    expected_y_train_scaled = y_train / 15.0
    # cond_train_scaled = original / 150
    expected_c_train_scaled = c_train / 150.0

    np.testing.assert_allclose(y_train_scaled, expected_y_train_scaled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_train_scaled, expected_c_train_scaled, rtol=1e-6, atol=1e-6)

    # Expected scaled val values
    expected_y_val_scaled = y_val / 15.0
    expected_c_val_scaled = c_val / 150.0

    np.testing.assert_allclose(y_val_scaled, expected_y_val_scaled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_val_scaled, expected_c_val_scaled, rtol=1e-6, atol=1e-6)


def test_minmax_scale_conditional_subset_normalize_only_target():
    """
    When only the target feature is in `normalize`, conditional features must
    remain unchanged.
    """
    y_train = np.array(
        [
            [[0.0], [10.0]],
            [[5.0], [15.0]],
        ],
        dtype=np.float32,
    )
    c_train = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    )
    y_val = np.array([[[20.0], [0.0]]], dtype=np.float32)
    c_val = np.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=np.float32)

    y_train_scaled, c_train_scaled, y_val_scaled, c_val_scaled = minmax_scale_conditional(
        y_train=y_train,
        c_train=c_train,
        y_val=y_val,
        c_val=c_val,
        target_feature="glucose",
        cond_features=["bolus", "carbs"],
        normalize=["glucose"],  # only target is scaled
    )

    # Target should be scaled to [0, 1] as before
    expected_y_train_scaled = y_train / 15.0
    expected_y_val_scaled = y_val / 15.0
    np.testing.assert_allclose(y_train_scaled, expected_y_train_scaled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y_val_scaled, expected_y_val_scaled, rtol=1e-6, atol=1e-6)

    # Conditional features must be unchanged
    np.testing.assert_array_equal(c_train_scaled, c_train)
    np.testing.assert_array_equal(c_val_scaled, c_val)


def test_minmax_scale_conditional_invalid_feature_name_raises():
    """
    If `normalize` contains a feature name not present in target+cond features,
    minmax_scale_features must raise a ValueError, propagated here.
    """
    y_train = np.random.rand(2, 3, 1)
    c_train = np.random.rand(2, 3, 2)
    y_val = np.random.rand(1, 3, 1)
    c_val = np.random.rand(1, 3, 2)

    with pytest.raises(ValueError):
        _ = minmax_scale_conditional(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            target_feature="glucose",
            cond_features=["bolus", "carbs"],
            normalize=["glucose", "unknown_feature"],
        )
