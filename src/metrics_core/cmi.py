from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D: (n,) -> (n,1)."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _standardize(
    *arrays: np.ndarray,
    jitter: float = 1e-6,
    rng: np.random.RandomState | None = None,
) -> list[np.ndarray]:
    """
    Standardize each array column-wise (zero mean, unit variance) and optionally add a tiny jitter.

    Notes
    -----
    - KSG estimators can become unstable with ties / identical samples (eps = 0, many zero distances).
      A small jitter helps break ties without changing the distribution meaningfully (if small enough).
    - Jitter scale is proportional to each column std after standardization.

    Parameters
    ----------
    arrays:
        Arrays with shape (n, d_i). All must share the same n.
    jitter:
        Relative jitter scale. Set 0 to disable.
    rng:
        RandomState for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Standardized (and jittered) arrays with the same shapes as inputs.
    """
    out: list[np.ndarray] = []
    rs = rng or np.random.RandomState()

    for X in arrays:
        X = np.asarray(X)
        if X.size == 0:
            out.append(X)
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        if jitter and Xs.size:
            eps = (np.std(Xs, axis=0, keepdims=True) + 1e-12) * float(jitter)
            noise = rs.normal(0.0, 1.0, size=Xs.shape) * eps
            Xs = Xs + noise

        out.append(Xs)

    return out


def _cmi_ksg_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    k: int = 10,
    metric: str = "chebyshev",
) -> float:
    """
    Estimate Conditional Mutual Information I(X;Y|Z) using a KSG / Frenzel–Pompe kNN estimator.

    Requirements
    ------------
    - X, Y, Z share the same number of rows n.
    - No NaNs: caller must drop rows with NaNs before calling.
    - Recommended preprocessing: standardize + small jitter (done upstream in df-level functions).
    - If k >= n, the estimator is not defined (returns NaN).

    Returns
    -------
    float
        Estimated CMI in bits (may be slightly negative due to finite-sample variance).
    """
    X = _as_2d(X)
    Y = _as_2d(Y)
    Z = _as_2d(Z)

    n = int(X.shape[0])
    if n == 0:
        return float("nan")

    if int(k) <= 0:
        raise ValueError("k must be positive.")

    if int(k) >= n:
        # Not enough samples to define the k-th neighbor distance.
        return float("nan")

    XZ = np.concatenate([X, Z], axis=1) if Z.size else X
    YZ = np.concatenate([Y, Z], axis=1) if Z.size else Y
    XYZ = np.concatenate([X, Y, Z], axis=1) if Z.size else np.concatenate([X, Y], axis=1)

    # k-th neighbor distance in joint space
    nn_joint = NearestNeighbors(metric=metric, n_neighbors=int(k) + 1).fit(XYZ)
    dists, _ = nn_joint.kneighbors(XYZ, n_neighbors=int(k) + 1)
    eps = dists[:, int(k)]
    tol = 1e-12

    def count_lt_eps(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return np.zeros(n, dtype=int)

        nn = NearestNeighbors(metric=metric).fit(mat)
        out = np.empty(n, dtype=int)

        # Radius counting is done per point; this is the usual direct implementation.
        for i in range(n):
            r = max(float(eps[i]) - tol, 0.0)
            if r <= 0.0:
                out[i] = 0
                continue
            ind = nn.radius_neighbors(mat[i : i + 1], radius=r, return_distance=False)[0]
            out[i] = int(len(ind) - 1)  # exclude self

        return out

    nxz = count_lt_eps(XZ)
    nyz = count_lt_eps(YZ)
    nz = count_lt_eps(Z) if Z.size else np.zeros(n, dtype=int)

    # KSG-CMI in nats, then convert to bits
    val_nats = digamma(int(k)) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
    return float(val_nats / np.log(2.0))


def _resolve_time_of_day_cols(df: pd.DataFrame, add_time_of_day: bool) -> list[str]:
    """Return time-of-day columns to include (only if present)."""
    if not add_time_of_day:
        return []
    return [c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns]


def compute_cmi_ksg(
    df: pd.DataFrame,
    *,
    target_col: str,
    candidate_cols: list[str],
    base_cols: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    k: int = 10,
    metric: str = "chebyshev",
    jitter: float = 1e-6,
    random_state: int = 0,
    min_samples: int = 100,
    clip_at_zero: bool = False,
) -> pd.DataFrame:
    """
    Compute KSG-CMI over multiple horizons:
        I(X_t ; Y_{t+h} | Z_t)

    Design choices
    --------------
    - Robust column handling: candidate/base columns missing from df are ignored (filtered out).
      If after filtering X is empty, cmi_bits is NaN for that horizon.

    Returns
    -------
    pd.DataFrame with one row per horizon:
      - horizon_min, n, cmi_bits, k, metric, unit
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")
    if int(freq_min) <= 0:
        raise ValueError("freq_min must be positive.")

    rng = np.random.RandomState(int(random_state))
    rows: list[dict[str, object]] = []

    tod_cols = _resolve_time_of_day_cols(df, bool(add_time_of_day))
    Z_cols = list(dict.fromkeys(list(base_cols) + tod_cols))
    X_cols = list(candidate_cols)

    # Keep only existing columns (robust behavior)
    Z_cols = [c for c in Z_cols if c in df.columns]
    X_cols = [c for c in X_cols if c in df.columns]

    horizons = sorted({int(h) for h in horizons_min})

    for h in horizons:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            # Skip invalid horizons (e.g. h=0)
            continue

        y = df[target_col].shift(-step).rename("_y_")
        needed = list(dict.fromkeys(Z_cols + X_cols))
        idx = df[needed].join(y).dropna().index
        n = int(len(idx))

        if n < int(min_samples) or len(X_cols) == 0:
            rows.append(
                {
                    "horizon_min": int(h),
                    "n": n,
                    "cmi_bits": np.nan,
                    "k": int(k),
                    "metric": str(metric),
                    "unit": "bits",
                }
            )
            continue

        X = df.loc[idx, X_cols].to_numpy()
        Z = df.loc[idx, Z_cols].to_numpy() if len(Z_cols) else np.empty((n, 0))
        Y = y.loc[idx].to_numpy().reshape(-1, 1)

        Xs, Ys, Zs = _standardize(X, Y, Z, jitter=float(jitter), rng=rng)
        val = _cmi_ksg_arrays(Xs, Ys, Zs, k=int(k), metric=str(metric))

        if bool(clip_at_zero) and np.isfinite(val):
            val = max(0.0, float(val))

        rows.append(
            {
                "horizon_min": int(h),
                "n": n,
                "cmi_bits": float(val) if np.isfinite(val) else np.nan,
                "k": int(k),
                "metric": str(metric),
                "unit": "bits",
            }
        )

    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)


def compute_cmi_ksg_decomposition(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    k: int = 10,
    metric: str = "chebyshev",
    jitter: float = 1e-6,
    random_state: int = 0,
    min_samples: int = 100,
    clip_at_zero: bool = True,
) -> pd.DataFrame:
    """
    Compute KSG-CMI AB decomposition over multiple horizons.

    For each horizon h:
      - I_total  = I([A,B] ; Y | base)
      - unique_A = I(A ; Y | base, B)
      - unique_B = I(B ; Y | base, A)
      - shared_raw = I_total - unique_A - unique_B
      - If clip_at_zero:
          synergy = max(shared_raw, 0)
          overlap = max(-shared_raw, 0)
        else:
          synergy = shared_raw
          overlap = 0

    Design choices
    --------------
    - Robust column handling: missing columns are ignored (filtered out).
      If A or B becomes empty after filtering, outputs are NaN for that horizon.
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")
    if int(freq_min) <= 0:
        raise ValueError("freq_min must be positive.")

    rng = np.random.RandomState(int(random_state))
    rows: list[dict[str, object]] = []

    tod_cols = _resolve_time_of_day_cols(df, bool(add_time_of_day))
    base_all = list(dict.fromkeys(list(base_cols) + tod_cols))

    # Keep only existing columns (robust behavior)
    base_all = [c for c in base_all if c in df.columns]
    A_cols = [c for c in features_A if c in df.columns]
    B_cols = [c for c in features_B if c in df.columns]

    needed_all = list(dict.fromkeys(base_all + A_cols + B_cols))
    horizons = sorted({int(h) for h in horizons_min})

    for h in horizons:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            continue

        y = df[target_col].shift(-step).rename("_y_")
        idx = df[needed_all].join(y).dropna().index
        n = int(len(idx))

        if n < int(min_samples) or len(A_cols) == 0 or len(B_cols) == 0:
            rows.append(
                {
                    "horizon_min": int(h),
                    "n": n,
                    "I_total": np.nan,
                    "unique_A": np.nan,
                    "unique_B": np.nan,
                    "shared_raw": np.nan,
                    "synergy": np.nan,
                    "overlap": np.nan,
                    "k": int(k),
                    "metric": str(metric),
                    "unit": "bits",
                }
            )
            continue

        Base = df.loc[idx, base_all].to_numpy() if len(base_all) else np.empty((n, 0))
        A = df.loc[idx, A_cols].to_numpy()
        B = df.loc[idx, B_cols].to_numpy()
        Y = y.loc[idx].to_numpy().reshape(-1, 1)

        Base_s, A_s, B_s, Y_s = _standardize(Base, A, B, Y, jitter=float(jitter), rng=rng)

        I_total = _cmi_ksg_arrays(np.concatenate([A_s, B_s], axis=1), Y_s, Base_s, k=int(k), metric=str(metric))
        unique_A = _cmi_ksg_arrays(A_s, Y_s, np.concatenate([Base_s, B_s], axis=1), k=int(k), metric=str(metric))
        unique_B = _cmi_ksg_arrays(B_s, Y_s, np.concatenate([Base_s, A_s], axis=1), k=int(k), metric=str(metric))

        shared_raw = float(I_total - unique_A - unique_B)

        if bool(clip_at_zero):
            synergy = max(0.0, shared_raw)
            overlap = max(0.0, -shared_raw)
        else:
            synergy = shared_raw
            overlap = 0.0

        rows.append(
            {
                "horizon_min": int(h),
                "n": n,
                "I_total": float(I_total) if np.isfinite(I_total) else np.nan,
                "unique_A": float(unique_A) if np.isfinite(unique_A) else np.nan,
                "unique_B": float(unique_B) if np.isfinite(unique_B) else np.nan,
                "shared_raw": float(shared_raw) if np.isfinite(shared_raw) else np.nan,
                "synergy": float(synergy) if np.isfinite(synergy) else np.nan,
                "overlap": float(overlap) if np.isfinite(overlap) else np.nan,
                "k": int(k),
                "metric": str(metric),
                "unit": "bits",
            }
        )

    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)
