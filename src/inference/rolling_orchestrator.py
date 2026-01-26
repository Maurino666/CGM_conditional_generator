import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from .orchestrator import InferenceOrchestrator

class RollingInferenceOrchestrator(InferenceOrchestrator):
    """
    Orchestrator specialized for 'Rolling' (Stateful) Inference.

    Unlike the standard InferenceOrchestrator, which treats every window as 
    an independent event, this class processes data subject-by-subject 
    in strict chronological order.

    It stitches temporal windows together to form a continuous sequence, 
    feeds them to the model maintaining the Hidden State (memory) between windows, 
    and then slices the output back for reconstruction.

    Key Features:
    - Patient-level isolation (state is reset between subjects).
    - Continuous Hidden State propagation (Chain of Days).
    - 'Leaky Identity' injection via `static_refresh_rate`.
    """

    def run(
            self,
            dfs: list[pd.DataFrame],
            seq_len: int,
            output_dir: Path | None = None,
            file_prefix: str | None = None,
            split_name: str = "Rolling_Inference",
            static_refresh_rate: float = 0.2
    ) -> list[pd.DataFrame]:
        """
        Executes the rolling inference pipeline.

        Args:
            dfs: list of input DataFrames (one per subject/session).
            seq_len: The length of the windows (e.g., 288 for 24h). 
                     This dictates the size of the rolling block.
            output_dir: Directory where CSV results will be saved.
            file_prefix: Prefix for generated filenames.
            split_name: Name used for logging purposes.
            static_refresh_rate: The 'Leaky Identity' factor (alpha).
                                 0.0 = Pure rolling (hidden state passed as-is).
                                 0.1-0.3 = Mixes 10-30% of static identity back into
                                           the hidden state at every window transition
                                           to prevent long-term drift.

        Returns:
            list[pd.DataFrame]: The reconstructed DataFrames containing synthetic data.
        """

        if self.verbose:
            print(f"\n[RollingOrchestrator] Starting Stateful Run: '{split_name.upper()}'")
            print(f"   > Sequence Length: {seq_len}")
            print(f"   > Static Refresh Rate: {static_refresh_rate}")

        generated_dfs = []

        # CRITICAL DIFFERENCE: 
        # We iterate through DataFrames (Subjects) ONE BY ONE.
        # We cannot batch mixed subjects together because we need to preserve 
        # the chronological order and hidden state continuity for a specific patient.
        iterator = tqdm(dfs, desc="Processing Subjects", disable=not self.verbose)

        for df in iterator:
            # -------------------------------------------------------
            # 1. Windowing (Subject Isolation)
            # -------------------------------------------------------
            # We reuse the existing builder to handle scaling/slicing.
            # shuffle=False is MANDATORY here to keep time continuity.
            # step=seq_len ensures no overlap (Chain of Days).
            window_split = self.builder.build_subset(
                dfs=[df],
                seq_len=seq_len,
                step=seq_len,
                shuffle=False,
                split_name="single_subj_temp"
            )

            # Skip if the dataframe was too short to produce even one window
            if len(window_split) == 0:
                continue

            # -------------------------------------------------------
            # 2. Continuous Tensor Extraction
            # -------------------------------------------------------
            # The builder gives us a DataLoader of separated windows.
            # We need to "stitch" them into one long sequence (1, Total_Time, Feat)
            # to perform the rolling generation.
            c_dyn_long, c_stat_ref = self._extract_continuous_tensors(window_split.loader)

            # -------------------------------------------------------
            # 3. Rolling Generation
            # -------------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                # We call the new 'generate_rolling' method on the model.
                # This handles the GRU state propagation and the static refresh logic.
                y_hat_long = self.model.generate_rolling(
                    cond_seq_long=c_dyn_long,
                    cond_static=c_stat_ref,
                    window_size=seq_len,
                    static_refresh_rate=static_refresh_rate
                )

            # -------------------------------------------------------
            # 4. Slicing & Reshaping for Reconstruction
            # -------------------------------------------------------
            # The Reconstructor expects a generic array of windows: (Total_Windows, Seq_Len, 1).
            # The Model returned a continuous line: (1, Total_Time, 1).

            # We reshape the long sequence back into windows.
            # Note: We rely on the fact that Total_Time = Num_Windows * Seq_Len
            # (enforced by the builder logic and _extract_continuous_tensors).
            y_hat_np = y_hat_long.view(-1, seq_len, 1).cpu().numpy().astype(np.float32)

            # Safety check: ensure we match the number of templates created by the builder.
            # (Rare edge case: if builder dropped a partial batch, reshape might be tricky,
            # but usually builder creates complete batches).
            if len(y_hat_np) > len(window_split.templates):
                y_hat_np = y_hat_np[:len(window_split.templates)]

            # -------------------------------------------------------
            # 5. Reconstruction (Reuse)
            # -------------------------------------------------------
            # We delegate the complex task of denormalization, index restoration, 
            # and metadata injection back to the original reconstructor.
            reconstructed_list = self.reconstructor.reconstruct(
                templates=window_split.templates,
                meta=window_split.metadata,
                y_hat_windows=y_hat_np,
            )

            generated_dfs.extend(reconstructed_list)

        # -------------------------------------------------------
        # 6. Persistence
        # -------------------------------------------------------
        if output_dir:
            safe_prefix = file_prefix if file_prefix else "Rolling"
            self._save_results(generated_dfs, output_dir, safe_prefix)

        return generated_dfs

    def _extract_continuous_tensors(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to unpack a Windowed DataLoader into continuous tensors.

        It takes a loader containing batches of windows (B, S, F) and 
        concatenates them along the time dimension to form a single 
        long sequence (1, Total_Time, F).

        Args:
            loader: The DataLoader from WindowBuilder.

        Returns:
            c_dyn_long: Tensor of shape (1, Total_Time, Dynamic_Dim)
            c_stat_ref: Tensor of shape (1, Static_Dim) - taken from the first window.
        """
        c_dyn_list = []
        c_stat_ref = None

        # Iterate over batches provided by the loader
        for batch in loader:
            batch = self._move_to_device(batch)

            # Expected batch structure: [target, dynamic_cond, static_cond]
            # Adjust indices if your dataset returns a different tuple structure
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, c_dyn, c_stat = batch
            else:
                raise ValueError(f"Rolling Orchestrator expects batch of len 3 (y, c, c_stat). Got {len(batch)}")

            c_dyn_list.append(c_dyn)

            # We capture the static variables from the very first batch 
            # (assuming they are constant for the patient)
            if c_stat_ref is None:
                c_stat_ref = c_stat[0:1]  # Take the first sample, keep dim (1, Static_Dim)

        # Concatenate all batches along the batch dimension first
        # c_dyn chunks are (Batch_Size, Seq_Len, Feat)
        full_seq_stacked = torch.cat(c_dyn_list, dim=0)  # Result: (Total_Windows, Seq_Len, Feat)

        # Flatten the Window and Sequence dimensions into a single Time dimension
        total_windows, seq_len, feat_dim = full_seq_stacked.shape
        c_dyn_long = full_seq_stacked.view(1, -1, feat_dim)  # Result: (1, Total_Time, Feat)

        return c_dyn_long, c_stat_ref