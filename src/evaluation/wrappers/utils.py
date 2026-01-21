from __future__ import annotations

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
