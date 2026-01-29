from __future__ import annotations

import pandas as pd
import numpy as np

from .base import BaseReconstructor
from windowing.utils import WindowMetadata


class FullSequenceReconstructor(BaseReconstructor):
    """
    Reconstructor for Full Sequence Generation (e.g., Diffusion Models).

    Assumptions:
    1. Input is a LIST of arrays, not a stacked tensor (lengths vary).
    2. There is a 1-to-1 mapping between a generated sequence and a template.
    3. No window overlap logic is required.
    """

    def reconstruct(
            self,
            templates: dict[str, pd.DataFrame],
            meta: list[WindowMetadata],
            generated_outputs: list[np.ndarray]
    ) -> list[pd.DataFrame]:
        """
        Reconstructs DataFrames from a list of variable-length arrays.

        Args:
            templates: Dict mapping subject_id -> Original DataFrame (for index/static cols).
            meta: Metadata list corresponding to the generated outputs.
            generated_outputs: List of arrays [ (L1, 1), (L2, 1), ... ].
                               Each array represents the generated target for one subject.

        Returns:
            List of reconstructed pandas DataFrames.
        """
        results = []

        # Iterate linearly through metadata and outputs
        for i, m in enumerate(meta):
            sid = m.subject_id

            # 1. Get the generated target for this subject
            # Shape: (Length, 1) or (Length,)
            y_hat = generated_outputs[i].flatten()

            # 2. Retrieve the template DataFrame
            template_norm_df = templates.get(sid)
            if template_norm_df is None:
                print(f"[Reconstructor] Warning: No template found for {sid}. Skipping.")
                continue

            # Ensure lengths match (clip template if generated is shorter, or vice versa)
            gen_len = len(y_hat)

            # 3. Denormalization
            if self.normalizer is not None:
                template_real_df = self.normalizer.inverse_transform([template_norm_df])[0]
                # We pass the array and the name of the column.
                # The normalizer handles the math using the specific min/max for this feature.
                # Note: We reshape to (Length, 1) because inverse_transform_array usually expects 2D
                final_y = self.normalizer.inverse_transform_array(
                    array=y_hat.reshape(-1, 1),
                    feature_name=self.cfg.target_col
                ).flatten()
            else:
                template_real_df = template_norm_df
                final_y = y_hat
            # 4. Build the Result DataFrame
            # We use the index from the template
            df_out = pd.DataFrame(index=template_real_df.index[:gen_len])

            # A. Copy Conditional Columns
            for col in self.cfg.cond_cols:
                if col in template_real_df.columns:
                    df_out[col] = template_real_df[col].iloc[:gen_len].values

            # B. Copy True Target (if configured)
            if self.cfg.include_true_target and self.cfg.target_col in template_real_df.columns:
                df_out[self.cfg.target_col] = template_real_df[self.cfg.target_col].iloc[:gen_len].values

            # C. Add Synthetic Target
            df_out[self.cfg.synth_col] = final_y

            # D. Restore Metadata
            df_out.attrs["unique_id"] = sid
            # Optional: Copy other attrs if needed

            results.append(df_out)

        return results