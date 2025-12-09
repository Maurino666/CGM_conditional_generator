import numpy as np
import pandas as pd
from typing import Literal
from statsmodels.api import OLS, add_constant


# TODO fix
from .metric_utils import build_lagged_view

# -----------------------------
# Core helpers for (semi)partial
# -----------------------------
def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """
    Return residuals of y after regressing it on X (with intercept).
    """
    if X.empty:
        # No controls: residual is just y centered on mean of intercept model
        return y - y.mean()
    Xc = add_constant(X, has_constant='add')
    m = OLS(y, Xc, missing='drop').fit()
    # Align indices used in the fit
    used_idx = m.model.data.row_labels
    return y.loc[used_idx] - m.predict(Xc)


def partial_or_semipartial_from_lagged(
    df_lagged: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_cols: list[str],
    mode: Literal["partial", "semipartial"] = "partial"
) -> float:
    """
    Compute partial or semipartial correlation on an already lagged dataframe.
    - partial: corr( resid(y|Z) , resid(x|Z) )
    - semipartial: corr( y , resid(x|Z) )
    """
    cols_needed = [y_col, x_col] + control_cols
    data = df_lagged[cols_needed].dropna()
    if data.empty:
        return np.nan

    y = data[y_col]
    x = data[x_col]
    Z = data[control_cols] if control_cols else pd.DataFrame(index=data.index)

    if mode == "partial":
        ry = _residualize(y, Z)
        rx = _residualize(x, Z)
        # Align indices just in case
        idx = ry.index.intersection(rx.index)
        return np.corrcoef(ry.loc[idx], rx.loc[idx])[0, 1]
    else:  # semipartial
        rx = _residualize(x, Z)
        idx = y.index.intersection(rx.index)
        return np.corrcoef(y.loc[idx], rx.loc[idx])[0, 1]


# -----------------------------
# Public API: raw df + lagging
# -----------------------------
def compute_partial_semipartial(
    df: pd.DataFrame,
    *,
    target_col: str,
    target_lags: list[int],
    exog_lags: dict[str, list[int]],
    x_feature: str,
    x_lag: int,
    mode: Literal["partial", "semipartial"] = "partial",
    dropna_after_lag: bool = True
) -> tuple[float, pd.DataFrame, dict[str, list[str]]]:
    """
    Compute partial or semipartial correlation for a *single* lagged predictor
    X = <x_feature>_lag<x_lag> against Y = <target_col> at time t,
    controlling for all the other lagged predictors (AR lags + other exogenous lags).

    Parameters
    ----------
    df : pd.DataFrame
        Raw, time-indexed dataframe (e.g., output of your fill_data()).
    target_col : str
        Target column name (e.g., "glucose").
    target_lags : list[int]
        Autoregressive lags for the target (e.g., [1..12]).
    exog_lags : dict[str, list[int]]
        Mapping of exogenous feature -> list of lags to build.
        Must include `x_feature` with `x_lag` inside if you want that column tested.
    x_feature : str
        Feature name to test (e.g., "insulin").
    x_lag : int
        Specific lag (in samples) of x_feature to test (e.g., 6 for ~30 minutes at 5-min sampling).
    mode : {"partial","semipartial"}, default "partial"
        Which coefficient to compute.
    dropna_after_lag : bool, default True
        Drop rows with NaN after lagging.

    Returns
    -------
    coeff : float
        Partial or semipartial correlation in [-1, 1] (np.nan if not computable).
    df_lagged : pd.DataFrame
        The lagged view used for the computation (for inspection or reuse).
    columns_info : dict
        {"y": <y_col>, "x": <x_col>, "controls": [list of control columns]}
        Helpful to log/report what was controlled for.

    Notes
    -----
    - Controls include: ALL target_lags and ALL exogenous lags EXCEPT the tested x_col.
    - This yields the *unique* linear effect of the chosen x_feature at the chosen lag.
    """
    # 1) Build lagged view once (single source of truth for time)
    df_lagged = build_lagged_view(
        df=df,
        target_col=target_col,
        target_lags=target_lags,
        exog_lags=exog_lags,
        dropna=dropna_after_lag
    )

    if df_lagged.empty:
        return np.nan, df_lagged, {"y": target_col, "x": f"{x_feature}_lag{x_lag}", "controls": []}

    # 2) Define columns: y, x (the tested lag), and controls (all others)
    y_col = target_col
    x_col = f"{x_feature}_lag{x_lag}"

    # Controls = all target AR lags + all exogenous lag cols except x_col
    ar_cols = [c for c in df_lagged.columns if c.startswith(f"{target_col}_lag")]
    exog_cols = []
    for feat, lags in exog_lags.items():
        for k in lags:
            col = f"{feat}_lag{k}"
            if col in df_lagged.columns and col != x_col:
                exog_cols.append(col)

    control_cols = ar_cols + exog_cols

    # 3) Compute partial or semipartial on the lagged view
    coeff = partial_or_semipartial_from_lagged(
        df_lagged=df_lagged,
        y_col=y_col,
        x_col=x_col,
        control_cols=control_cols,
        mode=mode
    )

    return coeff, df_lagged, {"y": y_col, "x": x_col, "controls": control_cols}
