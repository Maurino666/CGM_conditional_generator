# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
             base_cols: List[str],
             add_time_of_day: bool = True,
             extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
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

def make_walkforward_splits(n_samples: int, spec: TemporalCVSpec) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crea fold walk-forward: [train  ... ][gap][test] ripetuto,
    mantenendo min_train_size e test_size costanti. Nessuno shuffle.
    """
    splits = []
    total_needed = spec.min_train_size + spec.purge_gap + spec.test_size
    # Quante finestre complete ci stanno?
    # Lasciamo lo stesso test_size e scorriamo in avanti di test_size ad ogni fold.
    max_folds = (n_samples - spec.min_train_size) // (spec.test_size)
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
def delta_r2_cv_nonlinear(
    df: pd.DataFrame,
    target_col: str,
    base_cols: List[str],
    features_A: List[str],
    features_B: List[str],
    horizons_min: List[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    cv_spec: TemporalCVSpec = TemporalCVSpec(),
    random_state: int = 42,
    clip_shared_at_zero: bool = True,
) -> pd.DataFrame:
    """
    Per ogni orizzonte h (in minuti):
      - y = target spostato di -h (in step da freq_min)
      - modelli non lineari su: base, base+A, base+B, base+A+B
      - R² out-of-sample per ciascun fold
      - ΔR² condizionate: unique_A_CV, unique_B_CV, shared_CV
    Ritorna un DataFrame con medie sui fold e dettagli di base.
    """
    rng = np.random.RandomState(random_state)
    rows = []

    # colonne utilizzate in totale (per "fairness", stesse righe per tutti i modelli)
    needed_all = set(base_cols)
    if add_time_of_day:
        needed_all |= set([c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns])
    needed_all |= set(features_A) | set(features_B)
    needed_all = list(needed_all)

    for h in horizons_min:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            continue

        # y futuro
        y = df[target_col].shift(-step).rename("_y_")

        # indice "pulito": nessun NaN in base+A+B+target
        idx = df[needed_all].join(y).dropna().index
        df_h = df.loc[idx]  # mantieni l'ordine temporale

        # design matrices
        X_base   = _build_X(df_h, base_cols, add_time_of_day, extra_cols=[])
        X_baseA  = _build_X(df_h, base_cols, add_time_of_day, extra_cols=features_A)
        X_baseB  = _build_X(df_h, base_cols, add_time_of_day, extra_cols=features_B)
        X_baseAB = _build_X(df_h, base_cols, add_time_of_day, extra_cols=(features_A + features_B))
        y_h      = y.loc[idx]

        n = len(y_h)
        if n < (cv_spec.min_train_size + cv_spec.purge_gap + cv_spec.test_size):
            rows.append({
                "horizon_min": h, "n": n,
                "r2_base": np.nan, "r2_baseA": np.nan, "r2_baseB": np.nan, "r2_baseAB": np.nan,
                "unique_A_CV": np.nan, "unique_B_CV": np.nan, "shared_CV": np.nan,
                "n_folds": 0,
            })
            continue

        # purge gap: almeno "step" per evitare leakage tra y_t+h e righe vicine
        gap = max(cv_spec.purge_gap, step)
        spec = TemporalCVSpec(
            n_splits=cv_spec.n_splits,
            test_size=cv_spec.test_size,
            min_train_size=cv_spec.min_train_size,
            purge_gap=gap
        )
        splits = make_walkforward_splits(n, spec)

        r2_base_f, r2_baseA_f, r2_baseB_f, r2_baseAB_f = [], [], [], []

        for (tr, te) in splits:
            # estrazione fold
            Xb_tr, Xb_te   = X_base.iloc[tr],   X_base.iloc[te]
            XA_tr, XA_te   = X_baseA.iloc[tr],  X_baseA.iloc[te]
            XB_tr, XB_te   = X_baseB.iloc[tr],  X_baseB.iloc[te]
            XAB_tr, XAB_te = X_baseAB.iloc[tr], X_baseAB.iloc[te]
            y_tr, y_te     = y_h.iloc[tr],      y_h.iloc[te]

            # modelli (stessa random_state per riproducibilità)
            M_base   = make_regressor(random_state)
            M_baseA  = make_regressor(random_state)
            M_baseB  = make_regressor(random_state)
            M_baseAB = make_regressor(random_state)

            # fit
            M_base.fit(Xb_tr,  y_tr)
            M_baseA.fit(XA_tr, y_tr)
            M_baseB.fit(XB_tr, y_tr)
            M_baseAB.fit(XAB_tr, y_tr)

            # R² out-of-sample sul test
            r2_base_f.append(  r2_score(y_te, M_base.predict(Xb_te)) )
            r2_baseA_f.append( r2_score(y_te, M_baseA.predict(XA_te)) )
            r2_baseB_f.append( r2_score(y_te, M_baseB.predict(XB_te)) )
            r2_baseAB_f.append(r2_score(y_te, M_baseAB.predict(XAB_te)) )

        # medie sui fold
        r2b   = float(np.mean(r2_base_f))
        r2bA  = float(np.mean(r2_baseA_f))
        r2bB  = float(np.mean(r2_baseB_f))
        r2bAB = float(np.mean(r2_baseAB_f))

        delta_total = r2bAB - r2b
        unique_A = r2bAB - r2bB
        unique_B = r2bAB - r2bA
        shared   = delta_total - unique_A - unique_B
        if clip_shared_at_zero:
            shared = max(0.0, shared)

        rows.append({
            "horizon_min": h,
            "n": n,
            "n_folds": len(splits),
            "r2_base": r2b,
            "r2_baseA": r2bA,
            "r2_baseB": r2bB,
            "r2_baseAB": r2bAB,
            "unique_A_CV": unique_A,
            "unique_B_CV": unique_B,
            "shared_CV": shared,
            "r2_base_folds": r2_base_f,
            "r2_baseA_folds": r2_baseA_f,
            "r2_baseB_folds": r2_baseB_f,
            "r2_baseAB_folds": r2_baseAB_f,
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
