import numpy as np
import pandas as pd


def compute_agp(
    cgm: pd.Series,
    *,
    freq: str = "5min",
    min_days_per_bin: int = 5,
    clamp_range: tuple[float, float] | None = (39.0, 401.0),
    round_timestamps: bool = True,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """
    Compute Ambulatory Glucose Profile (AGP) statistics across days.

    Requirements / assumptions (core contract)
    ------------------------------------------
    Input
    - `cgm` must be a pandas Series of glucose values (typically mg/dL).
    - `cgm.index` must be a DatetimeIndex (timezone-aware or naive are both fine).
    - Time ordering is not strictly required, but recommended; duplicates are allowed.

    Time binning & day definition
    - AGP is computed by grouping samples by (date, time-of-day bin).
    - "Day" is defined by `DatetimeIndex.normalize()` (local midnight in the index timezone).
      If your index is timezone-aware, ensure it is in the desired local timezone before calling.
    - If `round_timestamps=True`, timestamps are rounded to the nearest `freq` boundary prior
      to binning. This controls the time-of-day grid alignment.

    Data cleaning
    - If `clamp_range` is provided, values outside [low, high] are treated as missing (NaN)
      before aggregation.
    - Missing data is allowed; percentiles are computed with `np.nanpercentile`.

    Coverage constraint
    - For each time-of-day bin, `n_days` counts the number of days with a non-NaN value
      contributing to that bin.
    - If `n_days < min_days_per_bin`, percentile columns for that bin are set to NaN
      (but `n_days` is kept).

    Aggregation within a day/bin
    - If multiple points map to the same (date, time-of-day bin) (due to rounding or duplicates),
      they are combined using `pivot_table(..., aggfunc=aggfunc)`; by default `mean`.

    Output
    - Returns a DataFrame indexed by a full 24h TimedeltaIndex grid at `freq`,
      with columns: ['p10','p25','p50','p75','p90','iqr','n_days'].
    - Units: same as `cgm` (typically mg/dL). Percentiles are per-bin across days.

    Notes
    - This function is intentionally "pure": it does not plot or save files.
      Plotting and per-subject artifact handling should be done at a higher level.
    - This function does not resample/interpolate the input time series; it only bins
      by time-of-day. If you need strict regular sampling, do it before calling.
    """
    if not isinstance(cgm.index, pd.DatetimeIndex):
        raise TypeError("agp_core requires `cgm` to have a DatetimeIndex.")

    s = cgm.copy()

    # Optional plausibility clamp
    if clamp_range is not None:
        lo, hi = clamp_range
        s = s.where((s >= lo) & (s <= hi))

    # Optionally round timestamps to align to a common grid
    if round_timestamps:
        s.index = s.index.round(freq)

    # Derive date and time-of-day (Timedelta since midnight)
    dates = s.index.normalize()
    tod = s.index - dates

    df = pd.DataFrame({"glucose": s.to_numpy(dtype=float), "date": dates, "tod": tod})

    # Pivot: rows=time-of-day bins, cols=days. Merge duplicates via aggfunc.
    mat = df.pivot_table(index="tod", columns="date", values="glucose", aggfunc=aggfunc)

    # Full 24h grid to ensure consistent index
    step = pd.to_timedelta(freq)
    full_tod = pd.timedelta_range(start="0s", end=pd.Timedelta("24h") - step, freq=freq)
    mat = mat.reindex(full_tod)

    values = mat.to_numpy()  # shape: (n_bins, n_days)
    n_days = np.sum(~np.isnan(values), axis=1).astype(int)

    # Percentiles across days per bin
    # Note: nanpercentile may warn if an entire row is NaN; we accept NaNs in output.
    p10 = np.nanpercentile(values, 10.0, axis=1)
    p25 = np.nanpercentile(values, 25.0, axis=1)
    p50 = np.nanpercentile(values, 50.0, axis=1)
    p75 = np.nanpercentile(values, 75.0, axis=1)
    p90 = np.nanpercentile(values, 90.0, axis=1)

    agp = pd.DataFrame(
        {
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "iqr": p75 - p25,
            "n_days": n_days,
        },
        index=mat.index,
    )

    # Enforce minimum-day coverage per bin (keep n_days)
    mask = agp["n_days"] < int(min_days_per_bin)
    if mask.any():
        agp.loc[mask, ["p10", "p25", "p50", "p75", "p90", "iqr"]] = np.nan

    agp.index.name = "time_of_day"
    agp = agp[["p10", "p25", "p50", "p75", "p90", "iqr", "n_days"]]
    return agp
