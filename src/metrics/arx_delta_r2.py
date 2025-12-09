from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def compute_arx_delta_r2(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    lag_minutes: list[int],
    horizon_min: int,
    add_time_of_day: bool = True,
    time_col: str | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
    min_samples: int = 200,
) -> dict:
    """
    Compute ARX ΔR² with time-series CV.
    Baseline: AR(target lags [+ time-of-day if add_time_of_day=True]).
    Augmented: Baseline + candidate_cols.
    Returns mean R² for baseline and augmented, ΔR², per-fold details, and counts.

    Assumptions that match your feature factory:
      - Target lags exist as f"{target_col}_lag_{m}m" for m in lag_minutes
      - Time-of-day features named 'tod_sin_24h' and 'tod_cos_24h' (optional)
      - All candidate_cols are contemporaneous features at time t (no leakage)
    """

    # Infer sampling step (minutes) safely
    if time_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provide time_col or use a DatetimeIndex.")
        t = df.index.to_series()
    else:
        t = pd.to_datetime(df[time_col])
        if t.isna().any():
            raise ValueError("time_col contains invalid datetimes.")
    dt_series = t.diff().dt.total_seconds().div(60.0).bfill()
    dt_min = float(dt_series.median())
    if not np.isfinite(dt_min) or dt_min <= 0:
        raise ValueError("Cannot infer a positive sampling step.")

    # Build y (future target at +h)
    h_steps = max(1, int(round(horizon_min / dt_min)))
    y = df[target_col].shift(-h_steps).rename("y_future")

    # Collect baseline columns present in df
    base_cols: list[str] = []
    for m in lag_minutes:
        col = f"{target_col}_lag_{m}m"
        if col in df.columns:
            base_cols.append(col)

    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns:
                base_cols.append(c)

    if len(base_cols) == 0:
        raise ValueError("No baseline columns found. Ensure target lags (and optionally tod_*) exist.")

    # Candidate columns: keep only those present
    cand_cols = [c for c in candidate_cols if c in df.columns]
    if len(cand_cols) == 0:
        raise ValueError("No candidate columns found in DataFrame.")

    # Assemble design matrices and align rows (same rows for fair comparison)
    X_base = df[base_cols]
    X_aug = df[base_cols + cand_cols]
    data_base = pd.concat([y, X_base], axis=1).dropna()
    data_aug  = pd.concat([y, X_aug ], axis=1).dropna()
    common_idx = data_base.index.intersection(data_aug.index)

    if len(common_idx) < min_samples:
        return {
            "r2_base_mean": np.nan, "r2_aug_mean": np.nan, "delta_r2_mean": np.nan,
            "r2_base_folds": [], "r2_aug_folds": [], "delta_folds": [],
            "n_samples": int(len(common_idx)),
            "n_features_base": len(base_cols),
            "n_features_aug": len(base_cols) + len(cand_cols),
            "used_cols_base": base_cols,
            "used_cols_aug": base_cols + cand_cols,
        }

    yv  = y.loc[common_idx].to_numpy()
    Xb  = X_base.loc[common_idx].to_numpy()
    Xa  = X_aug.loc[common_idx].to_numpy()

    # Time-series CV: standardize each fold and use Ridge for stability
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def fold_r2(X: np.ndarray) -> np.ndarray:
        scores = []
        for tr, te in tscv.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge",  Ridge(alpha=alpha)),
            ])
            model.fit(X[tr], yv[tr])
            pred = model.predict(X[te])
            scores.append(r2_score(yv[te], pred))
        return np.array(scores, dtype=float)

    r2b = fold_r2(Xb)
    r2a = fold_r2(Xa)
    delta = r2a - r2b

    return {
        "r2_base_mean": float(np.nanmean(r2b)),
        "r2_aug_mean":  float(np.nanmean(r2a)),
        "delta_r2_mean": float(np.nanmean(delta)),
        "r2_base_folds": r2b.tolist(),
        "r2_aug_folds":  r2a.tolist(),
        "delta_folds":   delta.tolist(),
        "n_samples": int(len(common_idx)),
        "n_features_base": len(base_cols),
        "n_features_aug":  len(base_cols) + len(cand_cols),
        "used_cols_base": base_cols,
        "used_cols_aug":  base_cols + cand_cols,
    }

def compute_arx_delta_r2_over_horizons(
        df: pd.DataFrame,
        target_col: str,
        candidate_cols: list[str],
        lag_minutes: list[int],
        horizon_min: list[int],
        add_time_of_day: bool = True,
        time_col: str | None = None,
        n_splits: int = 5,
        alpha: float = 1.0,
        min_samples: int = 200,
    )->pd.DataFrame:

    rows = []
    for h in horizon_min:
        res = compute_arx_delta_r2(
            df,
            target_col,
            candidate_cols,
            lag_minutes,
            h,
            add_time_of_day=add_time_of_day,
            time_col=time_col,
            n_splits=n_splits,
            alpha=alpha,
            min_samples=min_samples,
        )
        rows.append({
            "horizon_min": h,
            "r2_base": res["r2_base_mean"],
            "r2_aug": res["r2_aug_mean"],
            "delta_r2": res["delta_r2_mean"],
            "n_samples": res["n_samples"],
        })
    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

def arx_ab_test_same_sample(
    df,
    target_col: str,
    base_lag_minutes: list[int],
    horizon_min: int,
    candidate_cols_base: list[str],
    extra_cols: list[str],                 # e.g. ["Basal_present"] oppure ["Basal_filled"]
    add_time_of_day: bool = True,
    time_col: str | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> dict:
    """
    Run ARX ΔR² twice on the *same rows*: (A) without `extra_cols`, (B) with `extra_cols`.
    Returns both results and the incremental gain due to `extra_cols`.
    Requires that `arx_delta_r2_cv` is available in scope.
    """

    # Infer minutes step and build future target for mask
    if time_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provide time_col or use a DatetimeIndex.")
        t = df.index.to_series()
    else:
        t = pd.to_datetime(df[time_col])
        if t.isna().any():
            raise ValueError("time_col contains invalid datetimes.")

    dt_min = float(t.diff().dt.total_seconds().div(60.0).bfill().median())
    if not np.isfinite(dt_min) or dt_min <= 0:
        raise ValueError("Cannot infer a positive sampling step.")
    h_steps = max(1, int(round(horizon_min / dt_min)))
    y_future = df[target_col].shift(-h_steps)

    # Collect baseline cols
    base_cols = [c for m in base_lag_minutes for c in [f"{target_col}_lag_{m}m"] if c in df.columns]
    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns:
                base_cols.append(c)
    if not base_cols:
        raise ValueError("No baseline columns (target lags / tod_*) found.")

    # Build a *common* validity mask for both A and B
    cand_A = [c for c in candidate_cols_base if c in df.columns]
    cand_B = [c for c in (candidate_cols_base + extra_cols) if c in df.columns]
    union_cols = list(set(base_cols) | set(cand_B))  # B superset ensures common sample

    mask = y_future.notna()
    for c in union_cols:
        mask &= df[c].notna()
    df_same = df.loc[mask]

    # Run the metric twice on the *same* filtered df
    res_A = compute_arx_delta_r2(
        df_same, target_col,
        candidate_cols=cand_A,
        lag_minutes=base_lag_minutes,
        horizon_min=horizon_min,
        add_time_of_day=add_time_of_day,
        time_col=time_col,
        n_splits=n_splits,
        alpha=alpha,
        min_samples=50,
    )
    res_B = compute_arx_delta_r2(
        df_same, target_col,
        candidate_cols=cand_B,
        lag_minutes=base_lag_minutes,
        horizon_min=horizon_min,
        add_time_of_day=add_time_of_day,
        time_col=time_col,
        n_splits=n_splits,
        alpha=alpha,
        min_samples=50,
    )

    # Incremental gain strictly attributable to `extra_cols`
    gain = (res_B["delta_r2_mean"] if np.isfinite(res_B["delta_r2_mean"]) else np.nan) - \
           (res_A["delta_r2_mean"] if np.isfinite(res_A["delta_r2_mean"]) else np.nan)

    return {
        "without_extra": res_A,
        "with_extra": res_B,
        "incremental_gain_due_to_extra": float(gain),
        "n_rows_common": int(len(df_same)),
        "used_baseline_cols": base_cols,
        "candidate_cols_A": cand_A,
        "candidate_cols_B": cand_B,
    }

def plot_AB_arx_delta_r2(
    resA: pd.DataFrame,
    resB: pd.DataFrame,
    resAB: pd.DataFrame,
    name: str | None = "X",
    title: str | None = None,
    save_path: Path | None = None,
    annotate_values: bool | None = True,
):


    if resA is None or resB is None or resAB is None:
        print(f"[{name}] Missing one of A/B/AB results, skipping plot.")
        return

    if title is None:
        title = f"[{name}] ARX delta R2"

    fig, ax = plt.subplots(figsize=(6, 4))

    def plot_one(res_df, label, marker):
        ax.plot(res_df["horizon_min"], res_df["delta_r2"], marker=marker, label=label)
        if annotate_values:
            for x, y in zip(res_df["horizon_min"], res_df["delta_r2"]):
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom")

    plot_one(resA, "A: events + bolus flags", "o")
    plot_one(resB, "B: basal", "s")
    plot_one(resAB, "A+B: combined", "^")

    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("ΔR² (ARX − AR)")
    ax.set_title(f"[{title}] ARX ΔR² vs forecast horizon")
    ax.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close()