from typing import Optional, Tuple

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path


def compute_agp(
    cgm: pd.Series,
    freq: str = "5min",
    min_days_per_bin: int = 5,
    clamp_range: Optional[Tuple[float, float]] = (39.0, 401.0),
) -> pd.DataFrame:
    """
    Compute Ambulatory Glucose Profile (AGP) statistics across days.

    Steps:
    - Round timestamps to the target frequency (e.g., 5 min).
    - Map each timestamp to "time-of-day" (Timedelta since local midnight).
    - Pivot to a matrix with rows=time-of-day bins, columns=days.
    - Compute row-wise nan-percentiles across days: 10th, 25th, 50th, 75th, 90th.
    - Report also 'n_days' = number of non-NaN days contributing per bin.
    - Optionally clamp implausible CGM values.

    Parameters
    ----------
    cgm : pd.Series
        Glucose series with DatetimeIndex (mg/dL).
    freq : str
        Target grid frequency for time-of-day bins (e.g., "5min").
    min_days_per_bin : int
        Minimum number of days required to keep a bin; bins with fewer days are set to NaN.
    clamp_range : (low, high) or None
        If provided, values outside [low, high] are set to NaN before aggregation.

    Returns
    -------
    agp : pd.DataFrame
        Index: TimedeltaIndex from 00:00 to <24:00 at given freq.
        Columns: ['p10','p25','p50','p75','p90','iqr','n_days'].
    """
    s = cgm.copy()

    # Optional plausibility clamp
    if clamp_range is not None:
        lo, hi = clamp_range
        s = s.where((s >= lo) & (s <= hi))

    # Round timestamps to frequency to place points on a common grid
    s.index = s.index.round(freq)

    # Derive "time-of-day" as Timedelta since local midnight and the "date"
    dates = s.index.normalize()
    tod = (s.index - dates)  # Timedelta since midnight (00:00)

    df = pd.DataFrame({"glucose": s.values, "date": dates, "tod": tod})

    # Pivot to matrix: rows = time-of-day bins, columns = days
    # If multiple points fell in the same (date, tod) after rounding, mean will merge them.
    mat = df.pivot_table(index="tod", columns="date", values="glucose", aggfunc="mean")

    # Reindex to a complete 24h grid to ensure all bins exist
    full_tod = pd.timedelta_range(start="0s", end=pd.Timedelta("24h") - pd.to_timedelta(freq), freq=freq)
    mat = mat.reindex(full_tod)

    # Row-wise statistics across days
    # Use nanpercentile to tolerate gaps
    def row_percentile(a2d: np.ndarray, q: float) -> np.ndarray:
        return np.nanpercentile(a2d, q=q, axis=1)

    values = mat.to_numpy()  # shape: (n_bins, n_days)
    n_days = np.sum(~np.isnan(values), axis=1)

    p10 = row_percentile(values, 10.0)
    p25 = row_percentile(values, 25.0)
    p50 = row_percentile(values, 50.0)  # median
    p75 = row_percentile(values, 75.0)
    p90 = row_percentile(values, 90.0)

    agp = pd.DataFrame(
        {
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "iqr": (p75 - p25),
            "n_days": n_days.astype(int),
        },
        index=mat.index,
    )

    # Enforce minimum-day coverage per bin
    mask = agp["n_days"] < int(min_days_per_bin)
    agp.loc[mask, ["p10", "p25", "p50", "p75", "p90", "iqr"]] = np.nan

    # Ensure nice index/columns order
    agp.index.name = "time_of_day"
    agp = agp[["p10", "p25", "p50", "p75", "p90", "iqr", "n_days"]]

    return agp

def plot_agp(
        df_agp: pd.DataFrame,
        name : str | None = "AGP",
        title : str =  None,
        save_path : Path | None = None,

):
    if title is None:
        title = f"[{name}] AGP results."

    fig, ax = plt.subplots(figsize=(10, 4))
    # Convert TimedeltaIndex to hours for the x-axis
    x_hours = agp.index.total_seconds() / 3600.0

    # 10-90% band (light)
    ax.fill_between(x_hours, agp["p10"], agp["p90"], alpha=0.15, label="10–90%")

    # 25-75% band (darker)
    ax.fill_between(x_hours, agp["p25"], agp["p75"], alpha=0.25, label="25–75% (IQR)")

    # Median
    ax.plot(x_hours, agp["p50"], linewidth=2.0, label="Median")

    # Cosmetics
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time of day (hours)")
    ax.set_ylabel("Glucose (mg/dL)")
    if title:
        ax.set_title(title)

    # x ticks every 4 hours
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close(fig)