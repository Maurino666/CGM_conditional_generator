from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from data_management.normalization import Normalizer
# Imports from your project structure
from windowing.utils import WindowMetadata
from .strategies import ReconstructionStrategy, OverwriteStrategy, AverageStrategy


@dataclass(frozen=True)
class ReconstructionConfig:
    """
    Configuration for window -> DataFrame reconstruction.
    """
    target_col: str
    cond_cols: list[str]
    synth_col: str = "glucose_synth"
    include_true_target: bool = True


class WindowReconstructor:
    """
    Reconstructs full-length DataFrames from windowed model outputs.

    It supports different strategies (Overwrite vs Average) to handle
    overlapping windows correctly.
    """

    def __init__(
            self,
            cfg: ReconstructionConfig,
            strategy: str = "overwrite"  # "overwrite" or "average"
    ) -> None:
        self.cfg = cfg

        # Strategy Factory
        if strategy == "average":
            self.strategy: ReconstructionStrategy = AverageStrategy()
        elif strategy == "overwrite":
            self.strategy = OverwriteStrategy()
        else:
            raise ValueError(f"Unknown reconstruction strategy: {strategy}")

    def reconstruct(
            self,
            *,
            templates: dict[str, pd.DataFrame],
            meta: list[WindowMetadata],
            # c_windows: np.ndarray, # useless for now, but can be useful in the future
            y_hat_windows: np.ndarray,
            normalizer: Normalizer | None = None,
    ) -> list[pd.DataFrame]:
        """
        Main reconstruction loop.

        Args:
            templates: Dict mapping subject_id (str) to original DataFrame.
            meta: Metadata list matching the windows.
            # c_windows: Conditional input windows (N, seq, F).
            y_hat_windows: Generated target windows (N, seq, 1).
            normalizer: Fitted normalizer to de-normalize data.

        Returns:
            List of reconstructed DataFrames, sorted by subject ID.
        """
        # 1. Validation
        if len(meta) != len(y_hat_windows):
            raise ValueError("Length mismatch: Metadata vs Y_hat")
        #if len(meta) != len(c_windows):
        #    raise ValueError("Length mismatch: Metadata vs Conditional input")

        # 2. Denormalization (Optional but recommended BEFORE averaging)
        # It is mathematically cleaner to average real values than normalized ones.
        final_y_windows = y_hat_windows
        if normalizer is not None:
            print(f"   [Reconstructor] De-normalizing '{self.cfg.target_col}' before reconstruction...")
            final_y_windows = normalizer.inverse_transform_array(
                array=y_hat_windows,
                feature_name=self.cfg.target_col,
            )

        # 3. Initialize Buffers per Subject
        # We need a dedicated buffer set for each subject in the templates
        subject_buffers = {}
        for sid, df in templates.items():
            subject_buffers[sid] = self.strategy.initialize_buffers(len(df))

        # 4. Place Windows into Buffers
        for i, m in enumerate(meta):
            sid = m.subject_id
            start = m.start_index

            if sid not in subject_buffers:
                # This might happen if 'templates' is a subset (e.g. only Val)
                # but 'meta' contains everything. Usually an error.
                print(f"   [!] Warning: Window for subject {sid} has no template. Skipping.")
                continue

            # Delegate placement to the strategy
            self.strategy.place(
                buffers=subject_buffers[sid],
                window=final_y_windows[i],
                start_row=start
            )

        # 5. Finalize and Build DataFrames
        raw_dfs = []

        # Sort keys to ensure deterministic output order
        for sid in sorted(templates.keys()):
            df_template = templates[sid]
            buffers = subject_buffers[sid]

            # Compute final array (e.g. perform division for average)
            synth_array = self.strategy.finalize(buffers)

            # Create new DataFrame based on template index
            df_out = pd.DataFrame(index=df_template.index)

            # A. Copy Conditionals (Cloning from template)
            for col in self.cfg.cond_cols:
                if col in df_template.columns:
                    df_out[col] = df_template[col]
                else:
                    df_out[col] = np.nan

            # B. Copy True Target (if requested)
            if self.cfg.include_true_target:
                if self.cfg.target_col in df_template.columns:
                    df_out[self.cfg.target_col] = df_template[self.cfg.target_col]
                else:
                    df_out[self.cfg.target_col] = np.nan

            # C. Insert Synthetic Target
            # synth_array is (N, 1), we flatten to (N,)
            df_out[self.cfg.synth_col] = synth_array.flatten()

            # Add metadata for traceability
            df_out.attrs["unique_id"] = sid

            raw_dfs.append(df_out)

        if normalizer is not None:
            print(f"   [Reconstructor] Batch de-normalizing {len(raw_dfs)} subjects...")
            return normalizer.inverse_transform(raw_dfs)

        return raw_dfs