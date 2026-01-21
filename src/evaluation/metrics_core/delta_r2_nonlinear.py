from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# ---------------- Walk-forward CV spec ----------------
@dataclass(frozen=True)
class TemporalCVSpec:
    """
    Walk-forward CV parameters.

    n_splits:
      Maximum number of folds (actual folds depend on n_samples).
    test_size:
      Number of rows in each test block.
    min_train_size:
      Minimum training size for the first fold.
    purge_gap:
      Number of rows between train_end and test_start to reduce leakage.
      IMPORTANT: for forecasting, this should be >= horizon_steps.
    """
    n_splits: int = 5
    test_size: int = 1000
    min_train_size: int = 2000
    purge_gap: int = 0


def make_walkforward_splits(n_samples: int, spec: TemporalCVSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward folds:
      [ train ... ][ gap ][ test ] repeated forward

    No shuffle. Each subsequent fold moves forward by test_size.

    Returns list of (train_idx, test_idx).
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    max_folds = (n_samples - spec.min_train_size) // spec.test_size
    n_folds = max(1, min(spec.n_splits, max_folds))
    if n_folds < 1:
        return splits

    first_train_end = n_samples - (n_folds * spec.test_size + spec.purge_gap)
    first_train_end = max(first_train_end, spec.min_train_size)

    for k in range(n_folds):
        train_end = first_train_end + k * spec.test_size
        test_start = train_end + spec.purge_gap
        test_end = test_start + spec.test_size
        if test_end > n_samples:
            break
        tr = np.arange(0, train_end, dtype=int)
        te = np.arange(test_start, test_end, dtype=int)
        splits.append((tr, te))

    return splits


# --------------- Model factory protocol ----------------
class Regressor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...


RegressorFactory = Callable[[int], Regressor]


# ------------------- Design matrix helper -------------------
def build_design_matrix(
    df: pd.DataFrame,
    *,
    base_cols: list[str],
    add_time_of_day: bool = True,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Select baseline columns + optional time-of-day + optional extra columns.

    Core requirements:
    - Columns MUST already exist in df (no feature creation here).
    - Time-of-day columns (if used) are: 'tod_sin_24h', 'tod_cos_24h'.
    """
    cols = list(base_cols)

    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns and c not in cols:
                cols.append(c)

    if extra_cols:
        for c in extra_cols:
            if c in df.columns and c not in cols:
                cols.append(c)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    return df[cols]


# =========================
# Core 1: Nonlinear ΔR² (single horizon helper)
# =========================
def _compute_delta_r2_nonlinear_single_horizon(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    candidate_cols: list[str],
    horizon_min: int,
    freq_min: int = 5,
    add_time_of_day: bool = True,
    cv_spec: TemporalCVSpec = TemporalCVSpec(),
    regressor_factory: RegressorFactory,
    random_state: int = 42,
    min_samples: int | None = None,
) -> dict[str, object]:
    """
    Internal helper: out-of-sample ΔR² using a nonlinear regressor + walk-forward CV (single horizon).
    """
    step = int(round(horizon_min / float(freq_min)))
    if step <= 0:
        raise ValueError(f"horizon_min={horizon_min} with freq_min={freq_min} gives non-positive step.")

    # Union of required columns for a fair/common sample
    needed = set(base_cols) | set(candidate_cols)
    if add_time_of_day:
        needed |= {c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns}
    needed = list(needed)

    y_future = df[target_col].shift(-step).rename("_y_")
    Z = df[needed].join(y_future).dropna()
    if Z.empty:
        return {
            "horizon_min": int(horizon_min),
            "step": int(step),
            "n": 0,
            "n_folds": 0,
            "r2_base_mean": np.nan,
            "r2_aug_mean": np.nan,
            "delta_r2_mean": np.nan,
            "r2_base_folds": [],
            "r2_aug_folds": [],
            "used_cols_base": [],
            "used_cols_aug": [],
        }

    df_h = df.loc[Z.index]
    y_h = y_future.loc[Z.index]

    # Purge gap must be >= horizon step
    gap = max(int(cv_spec.purge_gap), int(step))
    spec = TemporalCVSpec(
        n_splits=cv_spec.n_splits,
        test_size=cv_spec.test_size,
        min_train_size=cv_spec.min_train_size,
        purge_gap=gap,
    )

    n = int(len(y_h))
    min_needed = spec.min_train_size + spec.test_size + spec.purge_gap
    if min_samples is not None:
        min_needed = max(min_needed, int(min_samples))

    if n < min_needed:
        return {
            "horizon_min": int(horizon_min),
            "step": int(step),
            "n": n,
            "n_folds": 0,
            "r2_base_mean": np.nan,
            "r2_aug_mean": np.nan,
            "delta_r2_mean": np.nan,
            "r2_base_folds": [],
            "r2_aug_folds": [],
            "used_cols_base": base_cols,
            "used_cols_aug": base_cols + candidate_cols,
        }

    Xb = build_design_matrix(df_h, base_cols=base_cols, add_time_of_day=add_time_of_day)
    Xa = build_design_matrix(df_h, base_cols=base_cols, add_time_of_day=add_time_of_day, extra_cols=candidate_cols)

    splits = make_walkforward_splits(n, spec)

    r2b_f: list[float] = []
    r2a_f: list[float] = []

    for tr, te in splits:
        Xb_tr, Xb_te = Xb.iloc[tr], Xb.iloc[te]
        Xa_tr, Xa_te = Xa.iloc[tr], Xa.iloc[te]
        y_tr, y_te = y_h.iloc[tr], y_h.iloc[te]

        m_base = regressor_factory(random_state)
        m_aug = regressor_factory(random_state)

        m_base.fit(Xb_tr, y_tr)
        m_aug.fit(Xa_tr, y_tr)

        r2b_f.append(float(r2_score(y_te, m_base.predict(Xb_te))))
        r2a_f.append(float(r2_score(y_te, m_aug.predict(Xa_te))))

    r2b = float(np.mean(r2b_f)) if r2b_f else np.nan
    r2a = float(np.mean(r2a_f)) if r2a_f else np.nan

    return {
        "horizon_min": int(horizon_min),
        "step": int(step),
        "n": n,
        "n_folds": int(len(splits)),
        "r2_base_mean": r2b,
        "r2_aug_mean": r2a,
        "delta_r2_mean": float(r2a - r2b) if np.isfinite(r2a) and np.isfinite(r2b) else np.nan,
        "r2_base_folds": r2b_f,
        "r2_aug_folds": r2a_f,
        "used_cols_base": Xb.columns.tolist(),
        "used_cols_aug": Xa.columns.tolist(),
    }


# =========================
# Public API: Nonlinear ΔR² (multi-horizon)
# =========================
def compute_delta_r2_nonlinear(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    candidate_cols: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    cv_spec: TemporalCVSpec = TemporalCVSpec(),
    regressor_factory: RegressorFactory,
    random_state: int = 42,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """
    Compute nonlinear out-of-sample ΔR² over multiple horizons.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame (one row per horizon)
      - folds: pd.DataFrame (long format: horizon_min, fold, r2_base, r2_aug, delta_r2)
      - used_cols_base: list[str]
      - used_cols_aug: list[str]
      - meta: dict[str, object]
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    # Enforce candidate existence here (wrappers may pre-filter, but the core can be strict)
    missing_cand = [c for c in candidate_cols if c not in df.columns]
    if missing_cand:
        raise ValueError(f"Missing candidate columns in df: {missing_cand}")

    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    # For introspection/debug: compute used columns once on a representative horizon (first one)
    # but keep correctness per-horizon in the row dicts.
    used_cols_base: list[str] = []
    used_cols_aug: list[str] = []

    for h in horizons_min:
        res = _compute_delta_r2_nonlinear_single_horizon(
            df,
            target_col=target_col,
            base_cols=base_cols,
            candidate_cols=candidate_cols,
            horizon_min=int(h),
            freq_min=int(freq_min),
            add_time_of_day=bool(add_time_of_day),
            cv_spec=cv_spec,
            regressor_factory=regressor_factory,
            random_state=int(random_state),
            min_samples=min_samples,
        )

        if not used_cols_base and isinstance(res.get("used_cols_base"), list):
            used_cols_base = list(res.get("used_cols_base", []))
        if not used_cols_aug and isinstance(res.get("used_cols_aug"), list):
            used_cols_aug = list(res.get("used_cols_aug", []))

        rows.append(
            {
                "horizon_min": int(res.get("horizon_min", h)),
                "step": int(res.get("step", int(round(int(h) / float(freq_min))))),
                "n": int(res.get("n", 0)),
                "n_folds": int(res.get("n_folds", 0)),
                "r2_base_mean": float(res.get("r2_base_mean", np.nan)),
                "r2_aug_mean": float(res.get("r2_aug_mean", np.nan)),
                "delta_r2_mean": float(res.get("delta_r2_mean", np.nan)),
            }
        )

        # folds
        r2b_f = res.get("r2_base_folds", [])
        r2a_f = res.get("r2_aug_folds", [])

        def to_float_list(x: object) -> list[float]:
            if not isinstance(x, list):
                return []
            out: list[float] = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float("nan"))
            return out

        rb = to_float_list(r2b_f)
        ra = to_float_list(r2a_f)
        n_folds = max(len(rb), len(ra))

        for k in range(n_folds):
            r2b = rb[k] if k < len(rb) else float("nan")
            r2a = ra[k] if k < len(ra) else float("nan")
            fold_rows.append(
                {
                    "horizon_min": int(res.get("horizon_min", h)),
                    "fold": int(k),
                    "r2_base": r2b,
                    "r2_aug": r2a,
                    "delta_r2": float(r2a - r2b) if np.isfinite(r2a) and np.isfinite(r2b) else float("nan"),
                }
            )

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values(["horizon_min", "fold"]).reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "folds": folds,
        "used_cols_base": used_cols_base,
        "used_cols_aug": used_cols_aug,
        "meta": {
            "freq_min": int(freq_min),
            "add_time_of_day": bool(add_time_of_day),
            "cv_spec": cv_spec,
            "random_state": int(random_state),
            "min_samples": int(min_samples) if min_samples is not None else None,
        },
    }


# ==========================================
# AB decomposition: single horizon helper
# ==========================================
def _compute_delta_r2_nonlinear_ab_decomposition_single_horizon(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizon_min: int,
    freq_min: int = 5,
    add_time_of_day: bool = True,
    cv_spec: TemporalCVSpec = TemporalCVSpec(),
    regressor_factory: RegressorFactory,
    random_state: int = 42,
    min_samples: int | None = None,
    clip_shared_at_zero: bool = True,
) -> dict[str, object]:
    """
    Internal helper: AB decomposition for nonlinear out-of-sample ΔR² on the SAME sample (single horizon).
    """
    step = int(round(horizon_min / float(freq_min)))
    if step <= 0:
        raise ValueError(f"horizon_min={horizon_min} with freq_min={freq_min} gives non-positive step.")

    A_cols = [c for c in features_A if c in df.columns]
    B_cols = [c for c in features_B if c in df.columns]
    AB_cols = list(dict.fromkeys(A_cols + B_cols))
    if not AB_cols:
        raise ValueError("No A/B columns found in df.")

    # Common sample mask for A/B/AB
    needed = set(base_cols) | set(AB_cols)
    if add_time_of_day:
        needed |= {c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns}
    needed = list(needed)

    y_future = df[target_col].shift(-step).rename("_y_")
    idx = df[needed].join(y_future).dropna().index
    df_same = df.loc[idx]

    details_A = _compute_delta_r2_nonlinear_single_horizon(
        df_same,
        target_col=target_col,
        base_cols=base_cols,
        candidate_cols=A_cols,
        horizon_min=horizon_min,
        freq_min=freq_min,
        add_time_of_day=add_time_of_day,
        cv_spec=cv_spec,
        regressor_factory=regressor_factory,
        random_state=random_state,
        min_samples=min_samples,
    )
    details_B = _compute_delta_r2_nonlinear_single_horizon(
        df_same,
        target_col=target_col,
        base_cols=base_cols,
        candidate_cols=B_cols,
        horizon_min=horizon_min,
        freq_min=freq_min,
        add_time_of_day=add_time_of_day,
        cv_spec=cv_spec,
        regressor_factory=regressor_factory,
        random_state=random_state,
        min_samples=min_samples,
    )
    details_AB = _compute_delta_r2_nonlinear_single_horizon(
        df_same,
        target_col=target_col,
        base_cols=base_cols,
        candidate_cols=AB_cols,
        horizon_min=horizon_min,
        freq_min=freq_min,
        add_time_of_day=add_time_of_day,
        cv_spec=cv_spec,
        regressor_factory=regressor_factory,
        random_state=random_state,
        min_samples=min_samples,
    )

    r2_base = float(details_AB.get("r2_base_mean", np.nan))
    r2_baseA = float(details_A.get("r2_aug_mean", np.nan))
    r2_baseB = float(details_B.get("r2_aug_mean", np.nan))
    r2_baseAB = float(details_AB.get("r2_aug_mean", np.nan))

    delta_total = r2_baseAB - r2_base if np.isfinite(r2_baseAB) and np.isfinite(r2_base) else np.nan
    unique_A = r2_baseAB - r2_baseB if np.isfinite(r2_baseAB) and np.isfinite(r2_baseB) else np.nan
    unique_B = r2_baseAB - r2_baseA if np.isfinite(r2_baseAB) and np.isfinite(r2_baseA) else np.nan
    shared = delta_total - unique_A - unique_B if all(np.isfinite([delta_total, unique_A, unique_B])) else np.nan
    if clip_shared_at_zero and np.isfinite(shared):
        shared = max(0.0, float(shared))

    return {
        "horizon_min": int(horizon_min),
        "step": int(step),
        "n": int(details_AB.get("n", len(df_same))),
        "n_folds": int(details_AB.get("n_folds", 0)),
        "r2_base": r2_base,
        "r2_baseA": r2_baseA,
        "r2_baseB": r2_baseB,
        "r2_baseAB": r2_baseAB,
        "delta_total": float(delta_total) if np.isfinite(delta_total) else np.nan,
        "unique_A": float(unique_A) if np.isfinite(unique_A) else np.nan,
        "unique_B": float(unique_B) if np.isfinite(unique_B) else np.nan,
        "shared": float(shared) if np.isfinite(shared) else np.nan,
        "details_A": details_A,
        "details_B": details_B,
        "details_AB": details_AB,
        "used_cols_baseline": base_cols,
        "used_cols_A": A_cols,
        "used_cols_B": B_cols,
        "used_cols_AB": AB_cols,
    }


# ==========================================
# Public API: Nonlinear AB decomposition (multi-horizon)
# ==========================================
def compute_delta_r2_nonlinear_ab_decomposition(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    cv_spec: TemporalCVSpec = TemporalCVSpec(),
    regressor_factory: RegressorFactory,
    random_state: int = 42,
    min_samples: int | None = None,
    clip_shared_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Compute nonlinear AB decomposition for out-of-sample ΔR² over multiple horizons.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame (one row per horizon) containing:
          r2_base, r2_baseA, r2_baseB, r2_baseAB,
          delta_total, unique_A, unique_B, shared,
          n, n_folds, step
      - details: dict[int, dict[str, object]] mapping horizon_min -> {details_A, details_B, details_AB, ...}
      - meta: dict[str, object]
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    rows: list[dict[str, Any]] = []
    details: dict[int, dict[str, Any]] = {}

    for h in horizons_min:
        res = _compute_delta_r2_nonlinear_ab_decomposition_single_horizon(
            df,
            target_col=target_col,
            base_cols=base_cols,
            features_A=features_A,
            features_B=features_B,
            horizon_min=int(h),
            freq_min=int(freq_min),
            add_time_of_day=bool(add_time_of_day),
            cv_spec=cv_spec,
            regressor_factory=regressor_factory,
            random_state=int(random_state),
            min_samples=min_samples,
            clip_shared_at_zero=bool(clip_shared_at_zero),
        )

        horizon_min = int(res.get("horizon_min", h))
        rows.append(
            {
                "horizon_min": horizon_min,
                "step": int(res.get("step", int(round(horizon_min / float(freq_min))))),
                "n": int(res.get("n", 0)),
                "n_folds": int(res.get("n_folds", 0)),
                "r2_base": float(res.get("r2_base", np.nan)),
                "r2_baseA": float(res.get("r2_baseA", np.nan)),
                "r2_baseB": float(res.get("r2_baseB", np.nan)),
                "r2_baseAB": float(res.get("r2_baseAB", np.nan)),
                "delta_total": float(res.get("delta_total", np.nan)),
                "unique_A": float(res.get("unique_A", np.nan)),
                "unique_B": float(res.get("unique_B", np.nan)),
                "shared": float(res.get("shared", np.nan)),
            }
        )

        details[horizon_min] = {
            "details_A": res.get("details_A", {}),
            "details_B": res.get("details_B", {}),
            "details_AB": res.get("details_AB", {}),
            "used_cols_baseline": res.get("used_cols_baseline", base_cols),
            "used_cols_A": res.get("used_cols_A", []),
            "used_cols_B": res.get("used_cols_B", []),
            "used_cols_AB": res.get("used_cols_AB", []),
        }

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "details": details,
        "meta": {
            "freq_min": int(freq_min),
            "add_time_of_day": bool(add_time_of_day),
            "cv_spec": cv_spec,
            "random_state": int(random_state),
            "min_samples": int(min_samples) if min_samples is not None else None,
            "clip_shared_at_zero": bool(clip_shared_at_zero),
        },
    }
