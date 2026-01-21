from __future__ import annotations

import pandas as pd
from data_prep.base_dataset import BaseDataset


class DataSplitter:
    """
    Unified Orchestrator for data splitting.

    It delegates the actual splitting logic to each BaseDataset instance
    (ensuring stratification by source) and then aggregates the results.
    """

    def __init__(
            self,
            val_ratio: float = 0.2,
            random_state: int = 42
    ) -> None:
        self.val_ratio = val_ratio
        self.random_state = random_state

    def split_data(
            self,
            datasets: list[BaseDataset],
            strategy: str
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Splits and Aggregates using the specified strategy.

        Injects the original subject ID into df.attrs['subject_id'] to preserve identity.
        """
        print(f"\n>>> [DataSplitter] Executing Strategy: {strategy.upper()}")

        all_train_dfs = []
        all_val_dfs = []

        for ds in datasets:
            ds_name = getattr(ds, 'name', 'Unknown')
            print(f"   - Processing {ds_name}...")

            # Calling dataset split
            train_sub, val_sub, train_ids, val_ids = ds.split(
                val_ratio=self.val_ratio,
                split_by=strategy,
                random_state=self.random_state
            )

            # Handling empty splits
            if not train_sub and not val_sub:
                print(f"     [!] WARNING: Dataset {ds_name} returned empty split.")
                continue

            # Pasting dataset id in df attributes
            for df, pid in zip(train_sub, train_ids):
                df.attrs["subject_id"] = str(pid)
                df.attrs["dataset_source"] = ds_name

            for df, pid in zip(val_sub, val_ids):
                df.attrs["subject_id"] = str(pid)
                df.attrs["dataset_source"] = ds_name

            # aggregation
            all_train_dfs.extend(train_sub)
            all_val_dfs.extend(val_sub)

            print(f"     -> Added {len(train_sub)} Train / {len(val_sub)} Val subjects")

        self._print_stats(all_train_dfs, all_val_dfs)
        return all_train_dfs, all_val_dfs

    def _print_stats(self, train: list[pd.DataFrame], val: list[pd.DataFrame]) -> None:
        n_train = len(train)
        n_val = len(val)
        rows_train = sum(len(df) for df in train)
        rows_val = sum(len(df) for df in val)

        print(f"   >>> Aggregated Split Complete.")
        print(f"   - Total Train: {n_train} subjects ({rows_train} rows)")
        print(f"   - Total Val:   {n_val} subjects ({rows_val} rows)")