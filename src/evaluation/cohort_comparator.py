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

    def plot_horizon_comparison(
            self,
            metric_config: dict[str, str],
            horizons: list[int],
            output_dir: Path | str
    ):
        """
        Generates line plots to visualize how metrics evolve or decay over prediction horizons.
        It compares Real vs Synthetic data for each specified metric family (e.g., ARX, Granger).

        The method performs a "wide-to-long" transformation internally to enable Seaborn
        to aggregate data across subjects and plot confidence intervals.

        Args:
            metric_config: A dictionary mapping the metric prefix (class name) to the specific suffix (field).
                           Example: {"arx_linear": "delta_r2_mean", "granger": "partial_r2_is"}
                           This allows reconstructing column names like 'arx_linear__h15m__delta_r2_mean_real'.
            horizons: List of integer horizons in minutes (e.g., [15, 30, 60, 120]).
                      These must match the horizons used during evaluation.
            output_dir: The directory where the plots will be saved.
        """
        out_path = Path(output_dir) / "horizon_dynamics"
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Iterate over each metric family defined in the config
        for prefix, suffix in metric_config.items():

            # Prepare a list to collect rows for the long-format DataFrame
            long_rows = []

            # 2. Identify available feature groups for stratification.
            #    If 'feature_set' is missing, treat the whole cohort as one group "All".
            feature_groups = self.paired_df["feature_set"].unique() if "feature_set" in self.paired_df.columns else [
                "All"]

            # 3. Iterate through groups to prepare plotting data
            for group in feature_groups:
                # Filter data for the current group
                if group == "All":
                    subset = self.paired_df
                else:
                    subset = self.paired_df[self.paired_df["feature_set"] == group]

                if subset.empty:
                    continue

                # 4. Extract data for each horizon
                for h in horizons:
                    # Construct the expected column names in the paired DataFrame.
                    # Naming convention: {prefix}__h{minutes}m__{suffix}_{source}
                    # Example: arx_linear__h30m__delta_r2_mean_real
                    col_base = f"{prefix}__h{h}m__{suffix}"
                    col_real = f"{col_base}_real"
                    col_synth = f"{col_base}_synth"

                    # Skip if columns are missing (e.g., if a specific horizon wasn't computed)
                    if col_real not in subset.columns or col_synth not in subset.columns:
                        continue

                    # Extract Real values and add to the list
                    vals_real = subset[col_real].dropna().values
                    for v in vals_real:
                        long_rows.append({
                            "feature_set": group,
                            "horizon": h,
                            "value": v,
                            "type": "Real"
                        })

                    # Extract Synthetic values and add to the list
                    vals_synth = subset[col_synth].dropna().values
                    for v in vals_synth:
                        long_rows.append({
                            "feature_set": group,
                            "horizon": h,
                            "value": v,
                            "type": "Synth"
                        })

            # Check if we actually collected any data
            if not long_rows:
                print(f"[Comparator] Warning: No data found for horizon plot of {prefix} (suffix={suffix}). Skipping.")
                continue

            # Create the Long DataFrame for Seaborn
            df_long = pd.DataFrame(long_rows)

            # 5. Generate one plot per feature group
            for group in df_long["feature_set"].unique():
                group_data = df_long[df_long["feature_set"] == group]

                plt.figure(figsize=(8, 5))

                # Plot Lines: Horizon (X) vs Value (Y), split by Type (Color/Style)
                # Seaborn automatically calculates the mean and 95% confidence interval (shaded band)
                sns.lineplot(
                    data=group_data,
                    x="horizon",
                    y="value",
                    hue="type",
                    style="type",
                    markers=True,
                    dashes=False,
                    palette=["#1f77b4", "#ff7f0e"]  # Blue for Real, Orange for Synth
                )

                # Formatting the plot
                # Truncate very long group names for the title
                short_group = (group[:40] + '..') if len(str(group)) > 40 else group

                plt.title(f"Horizon Dynamics: {prefix}\nMetric: {suffix} | Group: {short_group}")
                plt.xlabel("Forecast Horizon (minutes)")
                plt.ylabel(suffix.replace("_", " ").title())  # Prettify y-label (e.g., "Delta R2 Mean")
                plt.grid(True, alpha=0.3)

                # Force X-axis ticks to match exactly the requested horizons
                plt.xticks(horizons)

                # Save the figure
                # Create a safe filename by removing special characters from the group name
                safe_group = "".join([c if c.isalnum() else "_" for c in str(group)])[:50]
                fname = out_path / f"horizon_{prefix}__{suffix}__{safe_group}.png"

                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()