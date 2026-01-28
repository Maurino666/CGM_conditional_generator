from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import pandas as pd
from typing import Any

from models import BaseTrainableModule
from windowing.base import BaseDataBuilder
from reconstruction.base import BaseReconstructor

class BaseInferenceOrchestrator(ABC):
    """
    Abstract Base Class for Inference Orchestrators.

    It handles the shared infrastructure:
    1.  Initialization of core components (Model, Builder, Reconstructor).
    2.  Device management (moving tensors).
    3.  Persistence (saving DataFrames to CSVs).

    Subclasses must implement the `run()` method to define the specific
    execution flow (e.g., sliding window batching vs. full sequence streaming).
    """

    def __init__(
            self,
            model: BaseTrainableModule,
            builder: BaseDataBuilder,
            reconstructor: BaseReconstructor,
            device: torch.device,
            verbose: bool = True
    ):
        """
        Standard initialization logic.
        (Previously found in InferenceOrchestrator.__init__)
        """
        self.model = model
        self.builder = builder
        self.reconstructor = reconstructor
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def run(
            self,
            dfs: list[pd.DataFrame],
            **kwargs
    ) -> list[pd.DataFrame]:
        """
        Abstract contract for the pipeline execution.
        Subclasses must implement the logic to:
        1. Prepare data (Builder)
        2. Generate synthetic output (Model)
        3. Reconstruct DataFrames (Reconstructor)
        """
        pass

    def _save_results(
            self,
            dfs: list[pd.DataFrame],
            output_dir: Path,
            prefix: str,
    ) -> None:
        """
        Saves a list of DataFrames to CSV files.

        Logic location in original: InferenceOrchestrator._save_results
        Changes: None. This logic is generic and applies to any DataFrame.
        """
        csv_dir = output_dir / "csv_data"
        csv_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"   [Orchestrator] Saving {len(dfs)} CSV files to: {csv_dir}")

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
            print("   [Orchestrator] CSV saving complete.")

    def _move_to_device(self, batch: Any) -> Any:
        """
        Recursively moves tensors to the configured device.

        Logic location in original: InferenceOrchestrator._move_to_device
        Changes: None. Generic utility.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        return batch

    def _log(self, message: str) -> None:
        """Helper to print only if verbose is True."""
        if self.verbose:
            print(message)