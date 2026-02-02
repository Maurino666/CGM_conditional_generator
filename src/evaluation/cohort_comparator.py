from __future__ import annotations

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from matplotlib.pyplot import legend

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

        new_data = {}

        for col in numeric_cols:
            col_real = f"{col}_real"
            col_synth = f"{col}_synth"

            # Only compute delta if the metric exists in both (it should, usually)
            if col_real in paired.columns and col_synth in paired.columns:
                # Raw Difference: (Synth - Real)
                # Positive means Synth is higher; Negative means Synth is lower.
                diff_series = paired[col_synth] - paired[col_real]

                # Absolute Error: |Synth - Real|
                # Useful for magnitude of error regardless of direction.
                abs_err_series = diff_series.abs()

                # Saving in new_data to avoid fragmentation
                new_data[f"{col}_diff"] = diff_series
                new_data[f"{col}_abs_err"] = abs_err_series

        # Create a df from new_data and concat one-time
        if new_data:
            df_deltas = pd.DataFrame(new_data, index=paired.index)
            paired = pd.concat([paired, df_deltas], axis=1)

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

    def plot_metric_distributions(
            self,
            metrics: list[str],
            output_dir: Path,
    ):
        """
        Generates coupled plot of metric distributions.
        Creates a separated plot for every feature group.
        """

        out_path = Path(output_dir) / "distributions"
        out_path.mkdir(parents=True, exist_ok=True)

        # We only keep the metrics we are interested in
        cols_to_keep = ["subject_id", "feature_set"]
        for m in metrics:
            cols_to_keep.append(f"{m}_real")
            cols_to_keep.append(f"{m}_synth")

        subset = self.paired_df[cols_to_keep].copy()

        for metric in metrics:
            if f"{metric}_real" not in subset.columns:
                print("Skipped metric " + metric)
                continue

            real_data = subset[["feature_set", f"{metric}_real"]].rename(columns={f"{metric}_real": "value"})
            real_data["type"] = "Real"

            synth_data = subset[["feature_set", f"{metric}_synth"]].rename(columns={f"{metric}_synth": "value"})
            synth_data["type"] = "Synth"

            long_df = pd.concat([real_data, synth_data], axis=0, ignore_index=True)

            unique_groups = long_df["feature_set"].unique()

            # We plot every unique feature group
            for group in unique_groups:
                group_data = long_df[long_df["feature_set"] == group]

                plt.figure(figsize=(6, 5))
                sns.boxplot(
                    data=group_data,
                    x="type", y="value",
                    hue="type",
                    palette=["#1f77b4", "#ff7f0e"],
                    showfliers=False,
                    legend=False,
                )
                sns.stripplot(
                    data=group_data,
                    x="type",
                    y="value",
                    hue="type",
                    palette=["black", "black"],
                    alpha=0.3,
                    jitter=True,
                    legend=False,
                )

                plt.title(f"Metric: {metric}\nGroup: {group[:30]}...")
                plt.ylabel(metric)
                plt.grid(True, alpha=0.3)

                # Saving
                safe_group_name = "".join([c if c.isalnum() else "_" for c in str(group)])[:50]
                fname = out_path / f"{metric}__{safe_group_name}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()

    def plot_errors_by_group(self, metrics: list[str], output_dir: Path | str):
        """
        Generates plot of absolute errors grouped by feature group.
        """
        out_path = Path(output_dir) / "errors"
        out_path.mkdir(parents=True, exist_ok=True)

        cols_err = ["feature_set"] + [f"{m}_abs_err" for m in metrics]
        df_err = self.paired_df[cols_err].copy()


        for metric in metrics:
            col_name = f"{metric}_abs_err"
            if col_name not in df_err.columns:
                continue

            plt.figure(figsize=(10, 6))

            # Horizontal Bar Plot: Feature Group Y, Error X
            sns.boxplot(
                data=df_err,
                y="feature_set",
                x=col_name,
                hue="feature_set",
                orient="h",
                palette="viridis",
                legend=False,
            )

            plt.title(f"Absolute Error Distribution: {metric}")
            plt.xlabel(f"Absolute Error (|Synth - Real|)")
            plt.ylabel("Feature Configuration")
            plt.grid(True, alpha=0.3)

            fname = out_path / f"error_dist__{metric}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()