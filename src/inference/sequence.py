from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from .base import BaseInferenceOrchestrator

from models import BaseTrainableModule
from windowing import FullSequenceBuilder
from reconstruction import FullSequenceReconstructor


class SequenceInferenceOrchestrator(BaseInferenceOrchestrator):
    """
    Concrete Orchestrator for Full Sequence Inference (e.g., Diffusion Models).

    Logic Flow:
    1.  Uses FullSequenceBuilder to create a DataLoader with batch_size=1.
    2.  Iterates through the loader ONE subject at a time.
    3.  Collects results in a LIST (does not stack them, as lengths vary).
    4.  Uses FullSequenceReconstructor to process the list 1-to-1.
    """

    # --- TYPE HINTING OVERRIDE (Fixes IDE Warnings) ---
    builder: FullSequenceBuilder
    reconstructor: FullSequenceReconstructor

    # --------------------------------------------------

    def __init__(
            self,
            model: BaseTrainableModule,
            builder: FullSequenceBuilder,
            reconstructor: FullSequenceReconstructor,
            device: torch.device,
            verbose: bool = True
    ):
        """
        Note: We do NOT need seq_len here, as the builder extracts the full available length.
        """
        super().__init__(model, builder, reconstructor, device, verbose)

    def run(
            self,
            dfs: list[pd.DataFrame],
            output_dir: Path | None = None,
            file_prefix: str | None = None,
            split_name: str = "Inference",
            **kwargs
    ) -> list[pd.DataFrame]:

        self._log(f"\n[Orchestrator] Starting Sequence Run: '{split_name.upper()}'")

        # -----------------------------------------------------------
        # 1. Build Sequences (Batch Size = 1)
        # -----------------------------------------------------------
        # self.builder is correctly recognized as FullSequenceBuilder here
        seq_split = self.builder.build_sequences(
            dfs=dfs,
            split_name=split_name
        )

        if len(seq_split) == 0:
            self._log("   [!] Warning: No sequences extracted.")
            return []

        # -----------------------------------------------------------
        # 2. Generation (Stream Mode)
        # -----------------------------------------------------------
        self._log(f"   Running model generation (Stream Mode)...")

        # We collect outputs in a LIST, not a numpy stack
        generated_list = self._generate_stream(seq_split.loader)

        # -----------------------------------------------------------
        # 3. Reconstruction (1-to-1 Mapping)
        # -----------------------------------------------------------
        self._log(f"   Reconstructing {len(generated_list)} sequences...")

        # self.reconstructor is recognized as FullSequenceReconstructor
        synth_dfs = self.reconstructor.reconstruct(
            templates=seq_split.templates,
            meta=seq_split.metadata,
            generated_outputs=generated_list
        )

        # -----------------------------------------------------------
        # 4. Persistence
        # -----------------------------------------------------------
        if output_dir:
            safe_prefix = file_prefix if file_prefix is not None else "Gen"
            self._save_results(synth_dfs, output_dir, safe_prefix)

        return synth_dfs

    def _generate_stream(self, loader: torch.utils.data.DataLoader) -> list[np.ndarray]:
        """
        Iterates over the loader and generates outputs without stacking.
        Returns a list of numpy arrays, where each array corresponds to one subject.
        """
        self.model.eval()
        outputs = []

        iterator = loader
        if self.verbose:
            iterator = tqdm(loader, desc="   Inferencing", leave=False, file=sys.stdout)

        with torch.no_grad():
            for batch in iterator:
                # 1. Move to device
                batch = self._move_to_device(batch)

                # 2. Unpack (FullSequenceBuilder always returns Tuple[y, c])
                # Batch size is 1, but dimensions are (1, Length, Features)
                y_tensor, c_tensor = batch

                # 3. Generate
                # The model (Diffusion) should handle the (1, L, C) input correctly
                # and return (1, L, 1) output.
                out_tensor = self.model.generate(cond=c_tensor)

                # 4. Store
                # Remove the batch dimension (1) -> Result is (Length, 1)
                # We do NOT concatenate. We append to list.
                out_numpy = out_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                outputs.append(out_numpy)

        return outputs