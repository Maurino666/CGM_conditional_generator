"""
config_utils.py
===============
Utilities for loading and merging configuration dictionaries.
"""

from __future__ import annotations

import copy


def deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merge *overrides* into *base* (returns a new dict).

    - Dicts are merged recursively.
    - Lists and scalars in *overrides* **replace** the base value entirely.

    Example::

        base      = {"schema": {"target_col": "glucose", "static_cols": ["a","b","c"]}}
        overrides = {"schema": {"static_cols": ["a","b"]}}
        result    = {"schema": {"target_col": "glucose", "static_cols": ["a","b"]}}
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result