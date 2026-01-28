import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader
from .packs import SequenceSplit
from .base import  BaseDataBuilder
from .utils import extract_full_sequences


class VariableLengthDataset(Dataset):
    """Simple wrapper for list of (y, c) tuples."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return tensors directly
        return torch.tensor(self.data_list[idx][0]), torch.tensor(self.data_list[idx][1])


class FullSequenceBuilder(BaseDataBuilder):
    """
    Builder strategy: FULL SEQUENCES.

    Preserves the full length of each time-series.
    Does NOT stack tensors. Enforces batch_size=1.
    """

    def build_sequences(
            self,
            dfs: list[pd.DataFrame],
            split_name: str = "Inference"
    ) -> SequenceSplit:
        """
        Extracts full sequences for generation.
        """
        print(f"\n   [FullSequenceBuilder] Extracting sequences for '{split_name}'...")

        # 1. Logic Delegation (Using the new utility function)
        # Returns lists of arrays: [Arr1, Arr2], [Arr1, Arr2], ...
        list_y, list_c, metadata_list = extract_full_sequences(
            dfs=dfs,
            target_col=self.target_col,
            cond_cols=self.cond_cols,
            allow_target_nan=self.allow_target_nan
        )

        # 2. Template Mapping
        templates = {m.subject_id: dfs[i] for i, m in enumerate(metadata_list)}

        # 3. Dataset Creation
        # Using the custom VariableLengthDataset defined previously
        dataset_items = list(zip(list_y, list_c))
        dataset = VariableLengthDataset(dataset_items)

        # 4. Loader Creation
        # Specific constraints: batch_size=1, no shuffle
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        return SequenceSplit(
            loader=loader,
            metadata=metadata_list,
            templates=templates
        )