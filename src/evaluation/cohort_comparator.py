from __future__ import annotations

import pandas as pd
import numpy as np

from .types import EvaluationResult


class CohortComparator:
    """
    Orchestrates the comparison between Real and Synthetic evaluation results.

    It aligns subjects by ID, validates that their feature configurations (feature_sets)
    match, and computes pairwise differences for all metrics.
    """

    def __init__(self, real_res: EvaluationResult, synth_res: EvaluationResult):
        """
        Args:
            real_res: EvaluationResult object containing metrics for real data.
            synth_res: EvaluationResult object containing metrics for synthetic data.
        """
        self.real = real_res
        self.synth = synth_res

        # The master dataframe containing aligned rows and computed deltas
        self.paired_df: pd.DataFrame = self._align_and_compute_deltas()

    def _align_and_compute_deltas(self) -> pd.DataFrame:
        """
        Merges real and synthetic subject tables, validates consistency,
        and calculates difference metrics.

        Returns:
            A DataFrame with suffixes '_real' and '_synth', plus '_diff' and '_abs_err'
            columns for every numeric metric found in both sets.
        """
        # 1. Extract the per-subject DataFrames
        df_real = self.real.per_subject.copy()
        df_synth = self.synth.per_subject.copy()

        # 2. Set index to subject_id to ensure correct alignment during merge
        #    (Assuming subject_id is unique per DataFrame)
        if "subject_id" in df_real.columns:
            df_real.set_index("subject_id", inplace=True)
        if "subject_id" in df_synth.columns:
            df_synth.set_index("subject_id", inplace=True)

        # 3. Merge the DataFrames
        #    Inner join ensures we only compare subjects present in BOTH sets.
        #    Suffixes are applied automatically to overlapping column names.
        paired = df_real.join(
            df_synth,
            lsuffix="_real",
            rsuffix="_synth",
            how="inner"
        )

        if paired.empty:
            raise ValueError("Intersection of Real and Synth subjects is empty. Check subject_ids.")

        # 4. Consistency Check: Feature Sets
        #    We must ensure that Subject X in Real had the same available features
        #    as Subject X in Synth. If 'feature_set' was created by the Evaluator,
        #    it exists here.
        if "feature_set_real" in paired.columns and "feature_set_synth" in paired.columns:
            mismatches = paired[paired["feature_set_real"] != paired["feature_set_synth"]]
            if not mismatches.empty:
                raise ValueError(
                    f"Feature set mismatch detected for {len(mismatches)} subjects. "
                    "Real and Synthetic runs must use the same conditional columns per subject."
                )
            # If consistent, keep one 'feature_set' column for easier grouping later
            paired["feature_set"] = paired["feature_set_real"]

        # 5. Delta Computation
        #    Identify all numeric columns that exist in the original Real dataframe.
        #    (We exclude metadata columns like feature_set, id, etc.)
        numeric_cols = df_real.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_real = f"{col}_real"
            col_synth = f"{col}_synth"

            # Only compute delta if the metric exists in both (it should, usually)
            if col_real in paired.columns and col_synth in paired.columns:
                # Raw Difference: (Synth - Real)
                # Positive means Synth is higher; Negative means Synth is lower.
                paired[f"{col}_diff"] = paired[col_synth] - paired[col_real]

                # Absolute Error: |Synth - Real|
                # Useful for magnitude of error regardless of direction.
                paired[f"{col}_abs_err"] = paired[f"{col}_diff"].abs()

        return paired.reset_index()  # Bring subject_id back as a column

    def get_feature_groups(self) -> list[str]:
        """
        Returns the list of unique feature sets found in the cohort.
        Useful for iterating over groups for plotting.
        """
        if "feature_set" not in self.paired_df.columns:
            return []
        return sorted(self.paired_df["feature_set"].unique().tolist())

    def get_summary_by_group(self) -> pd.DataFrame:
        """
        Generates a summary table aggregated by feature_set.

        Returns:
            DataFrame with multi-index columns (Metric -> [mean_abs_err, std_abs_err])
        """
        if "feature_set" not in self.paired_df.columns:
            print("[Comparator] No 'feature_set' column found. Returning global summary.")
            return self.paired_df.describe().T

        # Identify all Absolute Error columns
        abs_err_cols = [c for c in self.paired_df.columns if c.endswith("_abs_err")]

        if not abs_err_cols:
            return pd.DataFrame()

        # Group by feature set and calculate Mean and Std of the Error
        grouped = self.paired_df.groupby("feature_set")[abs_err_cols].agg(["mean", "std"])

        return grouped