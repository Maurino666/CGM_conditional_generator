import pandas as pd
import numpy as np

def compute_mage(target: pd.Series,
                 dropna: bool = True,
                 value_range: tuple = (40, 400),
                 sd_threshold: float = 1.0,
                 smooth_window: int = 3,
                 min_separation: int = 1
    ) -> float:
    """
    Compute MAGE (Mean Amplitude of Glycemic Excursions).
    :param target: pandas Series with CGM values.
    :param dropna: whether to drop NaN values before processing.
    :param value_range: tuple (min, max) to filter target values.
    :param sd_threshold: threshold in terms of standard deviations to consider an excursion.
    :param smooth_window: window size for rolling median smoothing (1 = no smoothing).
    :param min_separation: minimum number of samples between consecutive extrema.
    :return: MAGE value or NaN if not computable.
    """
    # 0) Prepara la serie
    s = target.copy()
    if dropna:
        s = s.dropna()
    vmin, vmax = value_range
    s = s[(s >= vmin) & (s <= vmax)]
    if len(s) < 5:
        return np.nan

    # 1) Smussamento (rolling median)
    if smooth_window and smooth_window > 1:
        s = s.rolling(window=smooth_window, center=True, min_periods=1).median()
    vals = s.values.astype(float)

    # 2) Derivata e segni
    d1 = np.diff(vals)
    sign = np.sign(d1)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i-1] if sign[i-1] != 0 else 0

    # 3) Trova gli estremi locali
    extrema_idx = []
    for i in range(1, len(sign)):
        if sign[i] != sign[i-1]:
            extrema_idx.append(i)
    if len(extrema_idx) < 2:
        return np.nan

    # 4) Filtra per minima distanza
    if min_separation > 1:
        filtered = []
        last = -np.inf
        for idx in extrema_idx:
            if idx - last >= min_separation:
                filtered.append(idx)
                last = idx
        extrema_idx = filtered

    # 5) Costruisci alternanza picco ↔ valle
    def is_peak(i):
        return 0 < i < len(vals)-1 and vals[i] > vals[i-1] and vals[i] > vals[i+1]

    extrema = [i for i in extrema_idx if 0 < i < len(vals)-1]
    if len(extrema) < 2:
        return np.nan

    alt_extrema = [extrema[0]]
    for j in range(1, len(extrema)):
        prev = alt_extrema[-1]
        if is_peak(prev) == is_peak(extrema[j]):
            # stesso tipo: tieni quello più “forte”
            cand_keep = extrema[j] if abs(vals[extrema[j]] - vals[prev]) > 0 else prev
            alt_extrema[-1] = cand_keep
        else:
            alt_extrema.append(extrema[j])

    if len(alt_extrema) < 2:
        return np.nan

    # 6) Calcola le ampiezze
    amplitudes = [abs(vals[b] - vals[a]) for a, b in zip(alt_extrema[:-1], alt_extrema[1:])]
    amplitudes = np.array(amplitudes, dtype=float)
    if amplitudes.size == 0:
        return np.nan

    # 7) Applica la soglia basata su SD
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    theta = sd_threshold * sd
    qualified = amplitudes[amplitudes >= theta]
    if qualified.size == 0:
        return np.nan

    return float(np.mean(qualified))
