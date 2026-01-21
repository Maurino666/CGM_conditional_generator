import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from numpy.linalg import lstsq

from .metric_utils import build_lagged_view, build_future_targets, distance_correlation

#TODO fix

def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project y on columns of X (with intercept) and return residuals."""
    if X.ndim == 1:
        X = X[:, None]
    Xc = np.c_[np.ones(len(X)), X]
    beta, *_ = lstsq(Xc, y, rcond=None)
    y_hat = Xc @ beta
    return (y - y_hat).ravel()

def _circular_shifts(n: int, n_perm: int, min_shift: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw circular shift sizes in [1..n-1], avoiding tiny shifts if min_shift is set."""
    rng = rng or np.random.default_rng(0)
    if n <= 2:
        return np.array([1] * n_perm, dtype=int)
    lo = 1 if not min_shift else max(1, int(min_shift))
    hi = n - lo
    return rng.integers(low=lo, high=hi + 1, size=n_perm)

def partial_dcor_block(
    df: pd.DataFrame,
    *,
    target_col: str,
    target_lags: List[int],
    block_name: str,
    block_lags: List[int],
    horizons: List[int],
    use_common_row_mask: bool = True,
    full_exog_lags_for_mask: Optional[Dict[str, List[int]]] = None,
    n_perm: int = 200,
    max_samples: Optional[int] = None,  # cap rows after dropna (for speed/memory)
    step: int = 1,                      # downsample factor in time (1=no downsample)
    random_state: int = 0
) -> pd.DataFrame:
    """
    Compute partial dCor for one exogenous block across multiple horizons:
      dCor( X_block_lags , residuals_of(Y_future | AR_lags) )

    Returns a DataFrame with columns:
      ['block','horizon','dcor','pval','n_used']
    """
    rng = np.random.default_rng(random_state)

    # 1) Build lagged view for mask creation
    if use_common_row_mask and full_exog_lags_for_mask is not None:
        exog_for_mask = full_exog_lags_for_mask
    else:
        exog_for_mask = {block_name: block_lags}

    dfL = build_lagged_view(
        df=df, target_col=target_col,
        target_lags=target_lags,
        exog_lags=exog_for_mask,
        dropna=False
    )

    # 2) Assemble matrices
    # AR design
    ar_cols = [c for c in dfL.columns if c.startswith(f"{target_col}_lag")]
    X_ar = dfL[ar_cols]

    # Block design
    X_cols = [f"{block_name}_lag{k}" for k in block_lags if f"{block_name}_lag{k}" in dfL.columns]
    if len(X_cols) == 0:
        # Nothing to compute
        return pd.DataFrame([{"block": block_name, "horizon": h, "dcor": np.nan, "pval": np.nan, "n_used": 0} for h in horizons])

    X_block = dfL[X_cols]

    # Future targets
    Yf = build_future_targets(dfL[[target_col]], target_col=target_col, horizons=horizons)

    # 3) Row alignment + optional subsampling
    XY = X_ar.join(X_block, how="inner").join(Yf, how="inner")
    XY = XY.dropna()

    if step > 1:
        XY = XY.iloc[::step, :].copy()

    if (max_samples is not None) and (len(XY) > max_samples):
        XY = XY.iloc[-max_samples:, :]

    if XY.empty:
        return pd.DataFrame([{"block": block_name, "horizon": h, "dcor": np.nan, "pval": np.nan, "n_used": 0} for h in horizons])

    # 4) Compute residuals of Y_future on AR
    X_ar_used = XY[ar_cols].to_numpy()
    results = []

    for h in horizons:
        y = XY[f"{target_col}_lead{h}"].to_numpy().ravel()
        y_res = _residualize(y, X_ar_used)

        # Drop rows where residualization could produce NaN/Inf (paranoia)
        mask_ok = np.isfinite(y_res)
        Xb = XY[X_cols].to_numpy()[mask_ok]
        y_res_ok = y_res[mask_ok]
        n = len(y_res_ok)

        if n < 10 or Xb.shape[1] == 0:
            results.append({"block": block_name, "horizon": h, "dcor": np.nan, "pval": np.nan, "n_used": int(n)})
            continue

        # 5) Observed partial dCor
        d_obs = distance_correlation(Xb, y_res_ok.reshape(-1, 1))

        # 6) Permutation test via circular shifts of X (preserve autocorrelation)
        shifts = _circular_shifts(n=n, n_perm=n_perm, min_shift=max(1, int(0.1 * n)), rng=rng)
        greater = 0
        for s in shifts:
            Xb_shift = np.roll(Xb, shift=s, axis=0)
            d_perm = distance_correlation(Xb_shift, y_res_ok.reshape(-1, 1))
            greater += (d_perm >= d_obs)

        pval = (greater + 1) / (n_perm + 1)  # add-one smoothing
        results.append({"block": block_name, "horizon": h, "dcor": float(d_obs), "pval": float(pval), "n_used": int(n)})

    return pd.DataFrame(results)

def compute_partial_dcor(
    df: pd.DataFrame,
    *,
    target_col: str,
    target_lags: List[int],
    exog_lags: Dict[str, List[int]],
    horizons: List[int],
    use_common_row_mask: bool = True,
    n_perm: int = 200,
    max_samples: Optional[int] = None,
    step: int = 1,
    random_state: int = 0,
    also_all_block: bool = True
) -> pd.DataFrame:
    """
    Compute partial dCor for all exogenous blocks (and optional ALL) for given horizons.
    Returns a tidy DataFrame with one row per (block, horizon).
    """
    full_mask_lags = exog_lags if use_common_row_mask else None
    out = []
    for b, lags in exog_lags.items():
        df_b = partial_dcor_block(
            df=df,
            target_col=target_col,
            target_lags=target_lags,
            block_name=b,
            block_lags=lags,
            horizons=horizons,
            use_common_row_mask=use_common_row_mask,
            full_exog_lags_for_mask=full_mask_lags,
            n_perm=n_perm,
            max_samples=max_samples,
            step=step,
            random_state=random_state
        )
        out.append(df_b)

    if also_all_block:
        # Merge all exogenous columns into a single block "ALL"
        all_lags = {k: v for k, v in exog_lags.items()}
        df_all = partial_dcor_block(
            df=df,
            target_col=target_col,
            target_lags=target_lags,
            block_name="ALL",
            block_lags=sorted({x for vv in all_lags.values() for x in vv}),
            horizons=horizons,
            use_common_row_mask=use_common_row_mask,
            full_exog_lags_for_mask=full_mask_lags,
            n_perm=n_perm,
            max_samples=max_samples,
            step=step,
            random_state=random_state + 1
        )
        df_all["block"] = "ALL"
        out.append(df_all)

    return pd.concat(out, ignore_index=True)
