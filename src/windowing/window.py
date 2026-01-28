from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDataBuilder
from .utils import build_conditional_windows
from .packs import WindowSplit


class WindowBuilder(BaseDataBuilder):
    """
    Builder strategy: SLIDING WINDOWS.

    Transforms DataFrames into a dataset of fixed-length overlapping windows.
    Returns stacked Tensors suitable for standard batched training.
    """

    def __init__(
            self,
            max_missing_ratio: float = 0.0,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)
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

        has_static = len(self.static_indices) > 0
        if has_static: print(f"     -> [Fast-Loader] Static features detected.")

        # 3. Create Loader
        if count > 0:
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
            if has_static:
                c_dynamic_data = c_data[:, :, self.dynamic_indices]
                c_static_data = c_data[:, 0, self.static_indices]

                c_dynamic_tensor = torch.tensor(c_dynamic_data, dtype=torch.float32)
                c_static_tensor = torch.tensor(c_static_data, dtype=torch.float32)

                tensors_to_load = [y_tensor, c_dynamic_tensor, c_static_tensor]
            else:
                c_dynamic_data = c_data
                c_static_data = None
                c_tensor = torch.tensor(c_data, dtype=torch.float32)

                tensors_to_load = [y_tensor, c_tensor]

            num_workers = self.num_workers
            pin_memory = True if torch.cuda.is_available() else False


            if self.force_device is not None and self.force_device.type != 'cpu':
                # Load everything on device
                tensors_to_load = [t.to(self.force_device) for t in tensors_to_load]

                # Workers > 0 may create problems if tensors are already on vram
                num_workers = 0
                pin_memory = False
                print(f"     -> [Fast-Loader] Dataset loaded to {self.force_device}.")

            dataset = TensorDataset(*tensors_to_load)

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            print("     [!] Warning: Dataset is empty.")
            c_dynamic_data = np.empty((0, seq_len, len(self.dynamic_indices)), dtype=np.float32)
            c_static_data = np.empty((0, len(self.static_indices)), dtype=np.float32)
            loader = DataLoader([], batch_size=self.batch_size)

        # 4. Return the Split Object
        return WindowSplit(
            y=y_data,
            c_dynamic=c_dynamic_data,
            c_static=c_static_data if has_static else np.empty((count, 0)),

            dynamic_cols_indices=self.dynamic_indices if has_static else list(range(len(self.cond_cols))),
            static_cols_indices=self.static_indices,
            n_total_cond=len(self.cond_cols),

            loader=loader,
            metadata=metadata,
            templates=templates
        )