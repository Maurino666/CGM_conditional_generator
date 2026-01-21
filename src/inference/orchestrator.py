from __future__ import annotations

from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from windowing import WindowBuilder
from reconstruction import WindowReconstructor
from models import BaseTrainableModule


class InferenceOrchestrator:
    """
    Orchestrates the end-to-end inference pipeline for Generative Models.

    It encapsulates the following steps:
    1.  **Windowing**: Uses WindowBuilder to convert DataFrames into a WindowSplit
        (containing a batched DataLoader).
    2.  **Generation**: Iterates over the DataLoader, running the model on conditional
        inputs to produce synthetic targets.
    3.  **Reconstruction**: Delegates to WindowReconstructor to merge synthetic targets
        back into DataFrame format, handling denormalization and metadata.
    4.  **Persistence**: Saves the resulting DataFrames to CSV files.
    """

    def __init__(
            self,
            model: BaseTrainableModule,
            window_builder: WindowBuilder,
            reconstructor: WindowReconstructor,
            device: torch.device,
            verbose: bool = True
    ):
        """
        Args:
            model: The trained generative model (must implement `generate(condition)`).
            window_builder: Component responsible for slicing DataFrames and creating DataLoaders.
            reconstructor: Component responsible for converting tensors back to DataFrames.
            device: Torch device (CPU or CUDA).
            verbose: If True, prints progress logs to stdout.
        """
        self.model = model
        self.builder = window_builder
        self.reconstructor = reconstructor
        self.device = device
        self.verbose = verbose

    def run(
            self,
            dfs: list[pd.DataFrame],
            seq_len: int,
            output_dir: Path | None = None,
            file_prefix: str | None = None,
            scaling_params: dict[str, tuple[float, float]] | None = None,
            split_name: str = "Inference"
    ) -> list[pd.DataFrame]:
        """
        Executes the full inference pipeline on the provided DataFrames.

        Args:
            dfs: List of source DataFrames.
            seq_len: Length of extracted windows.
            output_dir: Directory where the resulting CSV files will be saved.
            file_prefix: Prefix for the generated filenames (e.g., 'val_subject').
            scaling_params: Params for de-normalization (passed to Reconstructor).
            split_name: Name of the current split (used for logging).

        Returns:
            list[pd.DataFrame]: The reconstructed DataFrames containing synthetic data.
        """
        if self.verbose:
            print(f"\n[InferenceOrchestrator] Starting run for split: '{split_name.upper()}'")

        # -----------------------------------------------------------
        # 1. Windowing (Non-Overlapping)
        # -----------------------------------------------------------
        # We enforce step=seq_len to ensure we generate every time point exactly once
        # without overlap. The builder handles dataset creation and batching.
        if self.verbose:
            print(f"   Building non-overlapping windows for {len(dfs)} DataFrames...")

        window_split = self.builder.build_subset(
            dfs=dfs,
            seq_len=seq_len,
            step=seq_len,  # Stride = Length -> No Overlap
            shuffle=False,  # Order must be preserved for reconstruction
            split_name=f"Gen_{split_name}"
        )

        if len(window_split) == 0:
            print(f"   [!] Warning: No windows created for {split_name}. Skipping.")
            return []

        # -----------------------------------------------------------
        # 2. Model Inference (Generation)
        # -----------------------------------------------------------
        if self.verbose:
            print(f"   Running model generation using DataLoader...")

        # We iterate over the loader provided by the split.
        y_hat_np = self._generate_from_loader(window_split.loader)

        # -----------------------------------------------------------
        # 3. Reconstruction
        # -----------------------------------------------------------
        if self.verbose:
            print(f"   Reconstructing {len(y_hat_np)} windows into DataFrames...")

        # The reconstructor handles denormalization internally if scaling_params are provided
        synth_dfs = self.reconstructor.reconstruct(
            templates=window_split.templates,
            meta=window_split.metadata,
            y_hat_windows=y_hat_np,
            scaling_params=scaling_params
        )

        # -----------------------------------------------------------
        # 4. Saving to Disk
        # -----------------------------------------------------------
        if output_dir:
            safe_prefix = file_prefix if file_prefix is not None else "Gen"
            self._save_results(synth_dfs, output_dir, safe_prefix)

        return synth_dfs

    def _generate_from_loader(self, loader: DataLoader) -> np.ndarray:
        """
        Iterates over the DataLoader provided by WindowBuilder.
        """
        self.model.eval()
        outputs = []

        iterator = loader
        if self.verbose:
            iterator = tqdm(loader, desc="   Inferencing", leave=False, file=sys.stdout)

        with torch.no_grad():
            for batch in iterator:
                # Batch from TensorDataset is a list: [y_tensor, c_tensor]
                # We move elements to device
                batch = self._move_to_device(batch)

                # Extract conditional input 'c' (index 1)
                # Structure validation logic:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    c = batch[1]
                else:
                    raise ValueError(f"Unexpected batch structure. Expected [y, c], got {type(batch)}")

                # Generate
                out = self.model.generate(c)

                # Handle Tuple Output (e.g., if model returns (X, Z))
                if isinstance(out, tuple):
                    y_hat = out[0]
                else:
                    y_hat = out

                # Move to CPU and store
                outputs.append(y_hat.detach().cpu().numpy().astype(np.float32))

        # Concatenate all batches along axis 0 -> (Total_Samples, Seq_Len, Feat)
        return np.concatenate(outputs, axis=0)

    def _save_results(
            self,
            dfs: list[pd.DataFrame],
            output_dir: Path,
            prefix: str,
    ) -> None:
        """
        Saves the list of DataFrames to CSV files using specific naming logic
        derived from DataFrame attributes.
        """
        csv_dir = output_dir / "csv_data"
        csv_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"   Saving {len(dfs)} CSV files to: {csv_dir}")

        for i, df in enumerate(dfs):
            # 1. Try retrieving 'unique_id' (most specific)
            unique_id = df.attrs.get("unique_id", None)

            if unique_id:
                filename = f"{prefix}_{str(unique_id)}.csv"
            else:
                # 2. Fallback to 'subject_id' + optional 'dataset_source'
                subject_id = df.attrs.get("subject_id", None)
                data_source = df.attrs.get("dataset_source", None)

                if subject_id:
                    if data_source:
                        filename = f"{prefix}_{str(data_source)}_{subject_id}.csv"
                    else:
                        filename = f"{prefix}_{str(subject_id)}.csv"
                else:
                    # 3. Final fallback: simple index
                    filename = f"{prefix}_{i:03d}.csv"

            save_path = csv_dir / filename
            df.to_csv(save_path, index=True)

        if self.verbose:
            print("   CSV saving complete.")

    def _move_to_device(self, batch):
        """Recursively moves tensors to the configured device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        return batch