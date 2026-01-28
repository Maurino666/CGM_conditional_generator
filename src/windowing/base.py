from __future__ import annotations
import torch

class BaseDataBuilder:
    """
    Base configuration class for Data Builders.

    Responsibilities:
    1. Validates input column configuration (Target, Conditional, Static).
    2. Computes feature indices (Static vs Dynamic).
    3. Stores infrastructure params (Batch size, Workers, Device).

    This class DOES NOT implement specific splitting logic (Windowing vs Full Sequence).
    """

    def __init__(
            self,
            target_col: str,
            cond_cols: list[str],
            static_cols: list[str] | None = None,
            batch_size: int = 64,
            num_workers: int = 0,
            allow_target_nan: bool = False,
            force_device: torch.device = None,
    ) -> None:
        # 1. Store Configuration
        self.target_col = target_col
        self.cond_cols = cond_cols
        self.static_cols = static_cols if static_cols else []

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_target_nan = allow_target_nan
        self.force_device = force_device

        # 2. Pre-calculate indices for feature separation
        # Useful for models that need to separate static inputs from dynamic ones.
        self.static_indices = [i for i, c in enumerate(cond_cols) if c in self.static_cols]
        self.dynamic_indices = [i for i, c in enumerate(cond_cols) if c not in self.static_cols]

        print(f"   [Builder Config] Initialized with Target: {target_col}")
        print(f"   [Builder Config] Static indices detected: {self.static_indices}")