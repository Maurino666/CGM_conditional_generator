from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import build_conditional_windows
from .packs import WindowSplit


class WindowBuilder:
    """
    Agnostic Builder: Transforms ANY list of DataFrames into a WindowSplit.
    It does not know about 'Train' or 'Validation' roles.
    """

    def __init__(
            self,
            target_col: str,
            cond_cols: list[str],
            batch_size: int = 64,
            num_workers: int = 0,
            max_missing_ratio: float = 0.05
    ) -> None:
        # Configuration shared across all splits
        self.target_col = target_col
        self.cond_cols = cond_cols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_missing_ratio = max_missing_ratio

    def build_subset(
            self,
            dfs: list[pd.DataFrame],
            seq_len: int,
            step: int,
            shuffle: bool = False,
            split_name: str = "dataset"  # Just for logging
    ) -> WindowSplit:
        """
        Processes a single list of DataFrames into windows.

        Args:
            dfs: List of Normalized DataFrames.
            seq_len: Window length for this specific split.
            step: Stride for this specific split.
            shuffle: Whether the DataLoader should shuffle (True for Train, False for Val/Test).
            split_name: Name for logging purposes.
        """
        print(f"\n   [WindowBuilder] Processing '{split_name}' (N={len(dfs)} subjects)...")
        print(f"     -> Params: seq_len={seq_len}, step={step}, shuffle={shuffle}")

        # 1. Build Templates
        templates = {}
        for i, df in enumerate(dfs):
            # Fallback ID if attribute is missing
            sid = str(df.attrs.get("subject_id", f"{split_name}_{i}"))
            templates[sid] = df

        # 2. Slice Windows
        y_data, c_data, metadata = build_conditional_windows(
            dfs=dfs,
            seq_len=seq_len,
            step=step,
            target_col=self.target_col,
            cond_cols=self.cond_cols,
            max_missing_ratio=self.max_missing_ratio
        )

        count = y_data.shape[0]
        print(f"     -> Generated {count} windows.")

        # 3. Create Loader
        if count > 0:
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
            c_tensor = torch.tensor(c_data, dtype=torch.float32)
            dataset = TensorDataset(c_tensor, y_tensor)

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
        else:
            print("     [!] Warning: Dataset is empty.")
            loader = DataLoader([], batch_size=self.batch_size)

        # 4. Return the Split Object
        return WindowSplit(
            y=y_data,
            c=c_data,
            loader=loader,
            metadata=metadata,
            templates=templates
        )