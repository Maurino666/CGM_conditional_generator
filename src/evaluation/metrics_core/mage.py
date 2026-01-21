import numpy as np
import pandas as pd


def compute_mage(
    values: pd.Series,
    *,
    dropna: bool = True,
    value_range: tuple[float, float] = (40.0, 400.0),
    sd_threshold: float = 1.0,
    smooth_window: int = 3,
    min_separation: int = 1,
) -> float:
    """
    Compute MAGE (Mean Amplitude of Glycemic Excursions) from a 1D glucose series.

    Requirements / assumptions (core contract)
    ------------------------------------------
    Input
    - `values` must be a 1D pandas Series of glucose measurements (typically mg/dL).
    - Index is ignored by this core implementation (time ordering must already be correct).
      If you pass time-indexed data, you must ensure it is already sorted by time.

    Pre-processing
    - If `dropna=True`, NaNs are removed.
    - `value_range=(lo, hi)` is used as a plausibility filter; values outside are removed.
    - If fewer than 5 samples remain after filtering, returns NaN.

    Method
    - Optional smoothing with centered rolling median of width `smooth_window` (>=2).
    - Local extrema are detected by sign changes in the first difference.
    - Extrema can be pruned by `min_separation` (in number of samples).
    - Excursion amplitudes are computed between alternating extrema.
    - Only amplitudes >= `sd_threshold * std(values)` are retained (std computed on the
      (possibly smoothed) filtered series with ddof=1).

    Output
    - Returns a float MAGE value (mean of qualified excursion amplitudes).
    - Returns NaN if the metric is not computable (not enough data, no extrema, sd<=0, etc.).

    Notes
    - This function is designed to be reusable outside the evaluator.
    - Any time-grid resampling / interpolation decisions must be handled outside this core.
    """
    s = values

    # Drop NaNs if requested
    if dropna:
        s = s.dropna()

    # Plausibility filtering
    lo, hi = value_range
    s = s[(s >= lo) & (s <= hi)]
    if len(s) < 5:
        return float("nan")

    # Optional rolling median smoothing
    if smooth_window > 1:
        s = s.rolling(window=smooth_window, center=True, min_periods=1).median()

    vals = s.to_numpy(dtype=float)

    # First differences and sign handling (carry forward zeros)
    d1 = np.diff(vals)
    sign = np.sign(d1)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1] if sign[i - 1] != 0 else 0

    # Find extrema indices: sign change points
    extrema_idx: list[int] = []
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            extrema_idx.append(i)

    if len(extrema_idx) < 2:
        return float("nan")

    # Enforce minimum separation between extrema (in samples)
    if min_separation > 1:
        filtered: list[int] = []
        last = -10**18
        for idx in extrema_idx:
            if idx - last >= min_separation:
                filtered.append(idx)
                last = idx
        extrema_idx = filtered

    # Keep only valid interior points
    extrema = [i for i in extrema_idx if 0 < i < len(vals) - 1]
    if len(extrema) < 2:
        return float("nan")

    def is_peak(i: int) -> bool:
        return vals[i] > vals[i - 1] and vals[i] > vals[i + 1]

    # Build an alternating sequence of peaks/valleys
    alt_extrema: list[int] = [extrema[0]]
    for j in range(1, len(extrema)):
        prev = alt_extrema[-1]
        cur = extrema[j]
        if is_peak(prev) == is_peak(cur):
            # Same type: keep the "stronger" one (heuristic)
            # (We keep `cur` if its value differs more from prev; otherwise keep prev.)
            if abs(vals[cur] - vals[prev]) > 0:
                alt_extrema[-1] = cur
        else:
            alt_extrema.append(cur)

    if len(alt_extrema) < 2:
        return float("nan")

    # Excursion amplitudes between consecutive alternating extrema
    amplitudes = np.abs(np.diff(vals[alt_extrema])).astype(float)
    if amplitudes.size == 0:
        return float("nan")

    # Threshold based on standard deviation
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")

    theta = sd_threshold * sd
    qualified = amplitudes[amplitudes >= theta]
    if qualified.size == 0:
        return float("nan")

    return float(np.mean(qualified))
