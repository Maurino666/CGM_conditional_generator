# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import r2_score
from pathlib import Path

# ------------------------- Model factory -------------------------
def make_regressor(random_state: int = 42):
    """
    Prova a usare LightGBM (più potente/veloce). Se non c'è, fallback a RandomForest.
    Mantieni settaggi "conservativi" per generalizzare bene.
    """
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            min_child_samples=40,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        # Fallback: scikit-learn
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

# ---------------------- Utility: design matrix -------------------
def _build_X(df: pd.DataFrame,
             base_cols: list[str],
             add_time_of_day: bool = True,
             extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Seleziona le colonne di base (lag) + opzionali di time-of-day + extra (A o B).
    Non crea nuove feature: assume che i lag e le TOD siano già presenti in df.
    """
    cols = list(base_cols)
    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns and c not in cols:
                cols.append(c)
    if extra_cols:
        for c in extra_cols:
            if c not in cols:
                cols.append(c)
    return df[cols]

# --------------- Utility: walk-forward temporal splits -----------
@dataclass
class TemporalCVSpec:
    n_splits: int = 5
    test_size: int = 1000          # numero di righe nel test per fold
    min_train_size: int = 2000     # per stabilità
    purge_gap: int = 0             # righe da "saltare" tra train e test per evitare leakage

def make_walkforward_splits(n_samples: int, spec: TemporalCVSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Crea fold walk-forward: [train  ... ][gap][test] ripetuto,
    mantenendo min_train_size e test_size costanti. Nessuno shuffle.
    """
    splits = []
    total_needed = spec.min_train_size + spec.purge_gap + spec.test_size
    # Quante finestre complete ci stanno?
    # Lasciamo lo stesso test_size e scorriamo in avanti di test_size ad ogni fold.
    max_folds = (n_samples - spec.min_train_size) // spec.test_size
    n_folds = max(1, min(spec.n_splits, max_folds))
    if n_folds < 1:
        return splits

    # Primo train finisce prima del primo test
    first_train_end = n_samples - (n_folds * spec.test_size + spec.purge_gap)
    # Garantisce train iniziale abbastanza lungo
    first_train_end = max(first_train_end, spec.min_train_size)

    for k in range(n_folds):
        train_end = first_train_end + k * spec.test_size
        test_start = train_end + spec.purge_gap
        test_end = test_start + spec.test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        splits.append((train_idx, test_idx))
    return splits

# ------------------- Core: ΔR² out-of-sample ---------------------
def compute_delta_r2_cv_nonlinear(
        df: pd.DataFrame,
        target_col: str,
        base_cols: list[str],
        candidate_cols: list[str],
        horizons_min: list[int],
        freq_min: int = 5,
        add_time_of_day: bool = True,
        cv_spec: TemporalCVSpec = TemporalCVSpec(),
        random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute out-of-sample ΔR² for a single candidate feature group.

    Models:
      baseline: base_cols (+ TOD)
      augmented: baseline + candidate_cols
    Method:
      - Future target: shift(-h)
      - Walk-forward CV, no shuffle
      - Model: LightGBM or RandomForest
    Output:
      R²_base, R²_aug, ΔR² per horizon (mean across folds)
    """
    rows = []

    needed_all = set(base_cols)
    if add_time_of_day:
        needed_all |= {c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns}
    needed_all |= set(candidate_cols)
    needed_all = list(needed_all)

    for h in horizons_min:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            continue

        # Build future target
        y = df[target_col].shift(-step).rename("_y_")

        # Drop NaNs in required features + y_future
        idx = df[needed_all].join(y).dropna().index
        df_h = df.loc[idx]
        y_h = y.loc[idx]

        n = len(y_h)
        min_needed = cv_spec.min_train_size + cv_spec.test_size + cv_spec.purge_gap
        if n < min_needed:
            rows.append({
                "horizon_min": h, "n": n, "n_folds": 0,
                "r2_base": np.nan, "r2_aug": np.nan, "delta_r2": np.nan,
                "r2_base_folds": [], "r2_aug_folds": [],
            })
            continue

        # Prevent leakage: purge gap ≥ forecast horizon
        gap = max(cv_spec.purge_gap, step)
        spec = TemporalCVSpec(
            n_splits=cv_spec.n_splits,
            test_size=cv_spec.test_size,
            min_train_size=cv_spec.min_train_size,
            purge_gap=gap,
        )
        splits = make_walkforward_splits(n, spec)

        Xb = _build_X(df_h, base_cols, add_time_of_day)
        Xa = _build_X(df_h, base_cols, add_time_of_day, extra_cols=candidate_cols)

        r2b_f, r2a_f = [], []

        for tr, te in splits:
            Xb_tr, Xb_te = Xb.iloc[tr], Xb.iloc[te]
            Xa_tr, Xa_te = Xa.iloc[tr], Xa.iloc[te]
            y_tr, y_te = y_h.iloc[tr], y_h.iloc[te]

            M_base = make_regressor(random_state)
            M_aug = make_regressor(random_state)

            M_base.fit(Xb_tr, y_tr)
            M_aug.fit(Xa_tr, y_tr)

            r2b_f.append(r2_score(y_te, M_base.predict(Xb_te)))
            r2a_f.append(r2_score(y_te, M_aug.predict(Xa_te)))

        r2b, r2a = np.mean(r2b_f), np.mean(r2a_f)

        rows.append({
            "horizon_min": h,
            "n": n,
            "n_folds": len(splits),
            "r2_base": float(r2b),
            "r2_aug": float(r2a),
            "delta_r2": float(r2a - r2b),
            "r2_base_folds": r2b_f,
            "r2_aug_folds": r2a_f,
        })

    return pd.DataFrame(rows)


def compute_delta_r2_cv_nonlinear_ab(
        df: pd.DataFrame,
        target_col: str,
        base_cols: list[str],
        features_A: list[str],
        features_B: list[str],
        horizons_min: list[int],
        freq_min: int = 5,
        add_time_of_day: bool = True,
        cv_spec: TemporalCVSpec = TemporalCVSpec(),
        random_state: int = 42,
        clip_shared_at_zero: bool = True,
) -> pd.DataFrame:
    """
    A/B comparison using compute_delta_r2_cv_nonlinear:
      base vs base+A,
      base vs base+B,
      base vs base+A+B,
    evaluated on the same valid rows.

    Output includes:
      r2_base, r2_baseA, r2_baseB, r2_baseAB
      unique_A_CV, unique_B_CV, shared_CV
    """
    rows = []

    needed_static = set(base_cols)
    if add_time_of_day:
        needed_static |= {c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns}
    needed_static |= set(features_A) | set(features_B)
    needed_static = list(needed_static)

    for h in horizons_min:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            continue

        y = df[target_col].shift(-step)
        idx = df[needed_static].join(y).dropna().index
        df_h = df.loc[idx]

        n = len(df_h)
        min_needed = cv_spec.min_train_size + cv_spec.test_size + cv_spec.purge_gap
        if n < min_needed:
            rows.append({
                "horizon_min": h, "n": n, "n_folds": 0,
                "r2_base": np.nan, "r2_baseA": np.nan,
                "r2_baseB": np.nan, "r2_baseAB": np.nan,
                "unique_A_CV": np.nan, "unique_B_CV": np.nan,
                "shared_CV": np.nan,
            })
            continue

        h_list = [h]

        res_A = compute_delta_r2_cv_nonlinear(df_h, target_col, base_cols, features_A,
                                      h_list, freq_min, add_time_of_day,
                                      cv_spec, random_state)
        res_B = compute_delta_r2_cv_nonlinear(df_h, target_col, base_cols, features_B,
                                      h_list, freq_min, add_time_of_day,
                                      cv_spec, random_state)
        res_AB = compute_delta_r2_cv_nonlinear(df_h, target_col, base_cols,
                                       features_A + features_B,
                                       h_list, freq_min, add_time_of_day,
                                       cv_spec, random_state)

        row_A, row_B, row_AB = res_A.iloc[0], res_B.iloc[0], res_AB.iloc[0]

        r2_base = float(row_AB["r2_base"])
        r2_baseA = float(row_A["r2_aug"])
        r2_baseB = float(row_B["r2_aug"])
        r2_baseAB = float(row_AB["r2_aug"])

        delta_total = r2_baseAB - r2_base
        unique_A = r2_baseAB - r2_baseB
        unique_B = r2_baseAB - r2_baseA
        shared = delta_total - unique_A - unique_B
        if clip_shared_at_zero:
            shared = max(0.0, shared)

        rows.append({
            "horizon_min": h,
            "n": n,
            "n_folds": int(row_AB["n_folds"]),
            "r2_base": r2_base,
            "r2_baseA": r2_baseA,
            "r2_baseB": r2_baseB,
            "r2_baseAB": r2_baseAB,
            "unique_A_CV": float(unique_A),
            "unique_B_CV": float(unique_B),
            "shared_CV": float(shared),
        })

    return pd.DataFrame(rows)


# -------------------------- Plot helper --------------------------
import matplotlib.pyplot as plt

def plot_delta_r2_cv(
        df_res: pd.DataFrame,
        name: str | None = "X",
        title: str = None,
        save_path : Path | None = None,
):
    if title is None:
        title =f"[{name}] Delta R2 results"

    h = df_res["horizon_min"].values
    plt.figure(figsize=(7, 4.5), dpi=100)
    plt.plot(h, df_res["r2_baseA"],  "-o", label="A (base+A)")
    plt.plot(h, df_res["r2_baseB"],  "-o", label="B (base+B)")
    plt.plot(h, df_res["r2_baseAB"], "-o", label="A+B (totale)")
    plt.plot(h, df_res["unique_A_CV"], "--s", label="unique A | B")
    plt.plot(h, df_res["unique_B_CV"], "--s", label="unique B | A")
    plt.plot(h, df_res["shared_CV"],   "--s", label="shared (≥0)")

    plt.title(title)
    plt.xlabel("Orizzonte (min)")
    plt.ylabel("R² out-of-sample / ΔR²")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close()
