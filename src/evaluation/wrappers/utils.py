from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.types import EvaluationConfig


def resolve_feature_group(df: pd.DataFrame, cfg: EvaluationConfig, group_key: str) -> list[str]:
    """
    Resolve a feature group from cfg.feature_groups and keep only columns present in df.
    """
    if cfg.feature_groups is None:
        return []
    cols = cfg.feature_groups.get(group_key, [])
    return [c for c in cols if c in df.columns]


def filter_valid_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """
    Filters the list of feature names, keeping only those that are present
    and 'valid' according to their mask.

    Logic:
    - If '{col}_mask' exists: keep 'col' only if sum(mask) > 0.
    - If '{col}_mask' does NOT exist: keep 'col' (assume valid/legacy data).
    - If 'col' is not in df: skip it.
    """
    valid = []
    for col in features:
        if col not in df.columns:
            continue

        mask_col = f"{col}_mask"
        if mask_col in df.columns:
            # Check if there is at least one valid sample (1.0)
            # We assume masks are 0.0 or 1.0.
            if df[mask_col].sum() > 0:
                valid.append(col)
        else:
            # No mask implies the column is fully valid
            valid.append(col)

    return valid


def restore_nans_from_masks_global(df: pd.DataFrame, cfg: EvaluationConfig) -> pd.DataFrame:
    """
    Globally restores NaN values for ALL columns defined in the configuration
    (target, synthetic, and conditional) based on their availability masks.

    This ensures that downstream metrics and feature engineering (like lags)
    never see '0.0' or fill values where the data was actually missing.

    Args:
        df: The input DataFrame containing data and '_mask' columns.
        cfg: The evaluation configuration defining which columns to check.

    Returns:
        A copy of the DataFrame with NaNs restored in invalid rows.
    """
    out = df.copy()

    # 1. Identify all columns that might have a corresponding mask
    cols_to_check = []

    # Target column
    if cfg.target_col in out.columns:
        cols_to_check.append(cfg.target_col)

    # Synthetic target column (if present)
    if cfg.comparison_target_col and cfg.comparison_target_col in out.columns:
        cols_to_check.append(cfg.comparison_target_col)

    # Conditional columns
    if cfg.cond_cols:
        # Only add those present in the current dataframe
        cols_to_check.extend([c for c in cfg.cond_cols if c in out.columns])

    # 2. Apply masks
    for col in cols_to_check:
        mask_col = f"{col}_mask"

        if mask_col in out.columns:
            # We assume binary masks: 1.0=Valid, 0.0=Invalid (missing).
            # We force NaN where the mask is 0.
            is_invalid = out[mask_col] == 0
            if is_invalid.any():
                out.loc[is_invalid, col] = np.nan

    return out