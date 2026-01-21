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
            max_missing_ratio: float = 0.0,
            allow_target_nan: bool = False,
            force_device: torch.device = None,
    ) -> None:
        # Configuration shared across all splits
        self.target_col = target_col
        self.cond_cols = cond_cols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_missing_ratio = max_missing_ratio
        self.allow_target_nan = allow_target_nan
        self.force_device = force_device

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
            # Get df information
            raw_id = str(df.attrs.get("subject_id", f"{split_name}_{i}"))
            source = df.attrs.get("dataset_source", None)

            # If source is present add it to the id
            if source:
                unique_id = f"{source}_{raw_id}"
            else:
                unique_id = raw_id

            # Safety check: if already exists does not override
            if unique_id in templates:
                print(f"     [Warning] ID Collision detected for {unique_id}. Appending index.")
                unique_id = f"{unique_id}_{i}"

            # update df attr
            df.attrs["unique_id"] = unique_id

            templates[unique_id] = df

        # 2. Slice Windows
        y_data, c_data, metadata = build_conditional_windows(
            dfs=dfs,
            seq_len=seq_len,
            step=step,
            target_col=self.target_col,
            cond_cols=self.cond_cols,
            max_missing_ratio=self.max_missing_ratio,
            allow_target_nan=self.allow_target_nan,
        )

        count = y_data.shape[0]
        print(f"     -> Generated {count} windows.")

        # 3. Create Loader
        if count > 0:
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
            c_tensor = torch.tensor(c_data, dtype=torch.float32)

            num_workers = self.num_workers
            pin_memory = True if torch.cuda.is_available() else False


            if self.force_device is not None and self.force_device.type != 'cpu':
                y_tensor = y_tensor.to(self.force_device)
                c_tensor = c_tensor.to(self.force_device)

                # Workers > 0 may create problems if tensors are already on vram
                num_workers = 0
                pin_memory = False
                print(f"     -> [Fast-Loader] Dataset loaded to {self.force_device}. Speed boost enabled 🚀")

            dataset = TensorDataset(y_tensor, c_tensor)

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
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