from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

# ---------- helper OLS ----------
def _ols_rss(X, y):
    # add intercept
    X_ = np.column_stack([np.ones(len(X)), X])
    # lstsq
    beta, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
    resid = y - X_ @ beta
    rss = float(resid.T @ resid)
    p = X_.shape[1]  # includes intercept
    return rss, p

def _build_X(df, lag_cols, add_time_of_day=True, extra_cols=None):
    parts = []
    parts.append(df[lag_cols])
    if add_time_of_day:
        tod_cols = [c for c in ("tod_sin_24h","tod_cos_24h") if c in df.columns]
        if tod_cols:
            parts.append(df[tod_cols])
    if extra_cols:
        parts.append(df[extra_cols])
    return pd.concat(parts, axis=1)

def _block_f_test(y, X_base, X_full):
    # OLS on reduced and full, then F-test and partial R^2
    rss_r, p_r = _ols_rss(X_base.values, y.values)
    rss_f, p_f = _ols_rss(X_full.values, y.values)
    n = len(y)
    k = p_f - p_r
    df_den = n - p_f
    if df_den <= 0 or k <= 0:
        return np.nan, np.nan, np.nan, k, df_den, n, rss_r, rss_f
    if rss_f <= 0:
        rss_f = 1e-12
    F = ((rss_r - rss_f) / k) / (rss_f / df_den)
    pval = f_dist.sf(F, k, df_den)
    partial_r2 = max(0.0, 1.0 - (rss_f / rss_r))
    return F, pval, partial_r2, k, df_den, n, rss_r, rss_f

# ---------- main: Granger multi-orizzonte con decomposizione ----------
def compute_granger_decomposition_over_horizons(
    df,
    target_col,
    lag_cols,
    horizons_min,
    features_A,
    features_B,
    add_time_of_day=True,
    min_samples=200,
    freq_min=5,
    match_n=True,   # usa le stesse righe per tutte le comparazioni
):
    """
    Per ogni orizzonte, calcola:
      - partial R^2 (in-sample) uncond.:  A, B, A+B
      - F-test e p-value per A, B, A+B
      - partial R^2 condizionali (η^2 parziale): unique_A = A|B, unique_B = B|A, e shared = (A+B) - unique_A - unique_B (clip >=0)
      - decomposizione additiva su base comune (ΔR^2): unique_A_delta, unique_B_delta, shared_delta tali che
            unique_A_delta + unique_B_delta + shared_delta == partial_r2_AB (a numerica).
    Ritorna un DataFrame con una riga per orizzonte.
    """
    out = []
    eps = 1e-12

    # colonne necessarie per il base
    base_needed = set(lag_cols)
    if add_time_of_day:
        base_needed |= set([c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns])

    for h in horizons_min:
        step = int(round(h / float(freq_min)))  # es. 15' con dati a 5' => 3 step
        if step <= 0:
            continue

        # target futuro
        y = df[target_col].shift(-step)

        # maschere di completezza (fairness same rows)
        base_cols = list(base_needed)
        XA_cols = list(features_A)
        XB_cols = list(features_B)
        XAB_cols = XA_cols + XB_cols  # assumiamo insiemi disgiunti

        if match_n:
            needed = set([target_col]) | set(base_cols) | set(XAB_cols)
            idx = df[list(needed)].join(y.rename("_y_")).dropna().index
        else:
            idx = df.index  # ogni test userà la propria maschera

        # design matrices
        X_base_all = _build_X(df.loc[idx], lag_cols, add_time_of_day, extra_cols=None).copy()
        XA_all = df.loc[idx, XA_cols]
        XB_all = df.loc[idx, XB_cols]
        XAB_all = df.loc[idx, XAB_cols]
        y_all = y.loc[idx]

        # helper per dropna local se non match_n
        def _dropna_local(yv, *Xs):
            Z = pd.concat([yv] + list(Xs), axis=1).dropna()
            yv2 = Z.iloc[:, 0]
            Xs2 = []
            cur = 1
            for X in Xs:
                cols = X.columns
                sl = Z.iloc[:, cur:cur + len(cols)]
                sl.columns = cols
                Xs2.append(sl)
                cur += len(cols)
            return yv2, Xs2

        if not match_n:
            yA, (X_base_A, XA)   = _dropna_local(y_all, X_base_all, XA_all)
            yB, (X_base_B, XB)   = _dropna_local(y_all, X_base_all, XB_all)
            yAB, (X_base_AB, XAB)= _dropna_local(y_all, X_base_all, XAB_all)
        else:
            yA, yB, yAB = y_all, y_all, y_all
            X_base_A = X_base_B = X_base_AB = X_base_all
            XA, XB, XAB = XA_all, XB_all, XAB_all

        # check min_samples
        if len(yAB) < min_samples:
            out.append({
                "horizon_min": h, "n": int(len(yAB)),
                "partial_r2_A": np.nan, "partial_r2_B": np.nan, "partial_r2_AB": np.nan,
                "unique_A": np.nan, "unique_B": np.nan, "shared": np.nan,
                "unique_A_delta": np.nan, "unique_B_delta": np.nan, "shared_delta": np.nan,
                "F_A": np.nan, "p_A": np.nan, "df_num_A": np.nan, "df_den_A": np.nan,
                "F_B": np.nan, "p_B": np.nan, "df_num_B": np.nan, "df_den_B": np.nan,
                "F_AB": np.nan, "p_AB": np.nan, "df_num_AB": np.nan, "df_den_AB": np.nan,
                "F_A_given_B": np.nan, "p_A_given_B": np.nan, "F_B_given_A": np.nan, "p_B_given_A": np.nan,
            })
            continue

        # --- Unconditional block tests ---
        F_A,  p_A,  pr2_A,  k_A,  dfden_A,  nA,  rss_red_A,  rss_full_A  = _block_f_test(yA,  X_base_A,  pd.concat([X_base_A,  XA],  axis=1))
        F_B,  p_B,  pr2_B,  k_B,  dfden_B,  nB,  rss_red_B,  rss_full_B  = _block_f_test(yB,  X_base_B,  pd.concat([X_base_B,  XB],  axis=1))
        F_AB, p_AB, pr2_AB, k_AB, dfden_AB, nAB, rss_red_AB, rss_full_AB = _block_f_test(yAB, X_base_AB, pd.concat([X_base_AB, XAB], axis=1))

        # --- Conditional uniques (η^2 parziale) ---
        # A | B : ridotto = (base+B), full = (base+A+B)
        _Ftmp, _ptmp, _pr2tmp, _k, _dfden, _n, rss_reduced_cond_B, rss_full_cond_B = _block_f_test(
            yAB, X_base_AB, pd.concat([X_base_AB, XB], axis=1)
        )
        # ATTENZIONE: dal test "base -> base+B", l'RSS di (base+B) è il *full* di quel test
        rss_base_plus_B = rss_full_cond_B
        unique_A = max(0.0, 1.0 - (rss_full_AB / rss_base_plus_B))

        # B | A : ridotto = (base+A), full = (base+A+B)
        _Ftmp2, _ptmp2, _pr2tmp2, _k2, _dfden2, _n2, rss_reduced_cond_A, rss_full_cond_A = _block_f_test(
            yAB, X_base_AB, pd.concat([X_base_AB, XA], axis=1)
        )
        rss_base_plus_A = rss_full_cond_A
        unique_B = max(0.0, 1.0 - (rss_full_AB / rss_base_plus_A))

        # F e p per i condizionali (stesso df_den del modello full AB)
        k_Acond = len(features_A)
        k_Bcond = len(features_B)
        den = max(1, dfden_AB)
        F_A_given_B = ((rss_base_plus_B - rss_full_AB) / k_Acond) / (rss_full_AB / den)
        p_A_given_B = f_dist.sf(F_A_given_B, k_Acond, den)
        F_B_given_A = ((rss_base_plus_A - rss_full_AB) / k_Bcond) / (rss_full_AB / den)
        p_B_given_A = f_dist.sf(F_B_given_A, k_Bcond, den)

        # shared condizionale (clip a >= 0 per numerica)
        shared = pr2_AB - unique_A - unique_B
        shared = np.nan if np.isnan(shared) else max(0.0, shared)

        # --- Decomposizione additiva (ΔR^2 su base comune) ---
        unique_A_delta = max(0.0, pr2_AB - pr2_B) if np.isfinite(pr2_AB) and np.isfinite(pr2_B) else np.nan
        unique_B_delta = max(0.0, pr2_AB - pr2_A) if np.isfinite(pr2_AB) and np.isfinite(pr2_A) else np.nan
        shared_delta = pr2_A + pr2_B - pr2_AB if all(map(np.isfinite, [pr2_A, pr2_B, pr2_AB])) else np.nan
        if np.isfinite(shared_delta):
            shared_delta = max(0.0, shared_delta)
            # ripulisci micro errori di somma
            s = unique_A_delta + unique_B_delta + shared_delta
            if np.isfinite(s) and np.isfinite(pr2_AB) and abs(s - pr2_AB) < 1e-10:
                pass  # ok
            # se per numerica s > AB di un pelo, ridistribuisci sullo shared
            if np.isfinite(s) and np.isfinite(pr2_AB) and s - pr2_AB > 0 and s - pr2_AB < 1e-8:
                shared_delta = max(0.0, shared_delta - (s - pr2_AB))

        out.append({
            "horizon_min": h, "n": int(len(yAB)),
            # partial R^2 uncond.
            "partial_r2_A": pr2_A, "partial_r2_B": pr2_B, "partial_r2_AB": pr2_AB,
            # condizionali (η^2 parziale)
            "unique_A": unique_A, "unique_B": unique_B, "shared": shared,
            # additivi (ΔR^2 su base)
            "unique_A_delta": unique_A_delta, "unique_B_delta": unique_B_delta, "shared_delta": shared_delta,
            # F-test uncond.
            "F_A": F_A, "p_A": p_A, "df_num_A": k_A, "df_den_A": dfden_A,
            "F_B": F_B, "p_B": p_B, "df_num_B": k_B, "df_den_B": dfden_B,
            "F_AB": F_AB, "p_AB": p_AB, "df_num_AB": k_AB, "df_den_AB": dfden_AB,
            # F-test cond.
            "F_A_given_B": F_A_given_B, "p_A_given_B": p_A_given_B,
            "F_B_given_A": F_B_given_A, "p_B_given_A": p_B_given_A,
        })

    return pd.DataFrame(out)


# ---------- plotting di supporto ----------
import matplotlib.pyplot as plt

def plot_granger_decomposition(
        df_res : pd.DataFrame,
        name : str | None = "X",
        title : str | None = None,
        save_path: Path | None = None,
):
    if title is None:
        title =f"[{name}] Delta R2 results"

    fig, ax = plt.subplots(figsize=(7,4.5))
    x = df_res["horizon_min"].values
    ax.plot(x, df_res["partial_r2_A"], marker="o", label="A (bolo/eventi)")
    ax.plot(x, df_res["partial_r2_B"], marker="o", label="B (basal)")
    ax.plot(x, df_res["partial_r2_AB"], marker="o", label="A+B (totale)")

    # opzionale: evidenzia unique vs shared
    ax.plot(x, df_res["unique_A"], marker="s", linestyle="--", label="unique A | B")
    ax.plot(x, df_res["unique_B"], marker="s", linestyle="--", label="unique B | A")

    ax.set_xlabel("Orizzonte (min)")
    ax.set_ylabel("partial R² (in-sample)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close()

def plot_AB_granger_decomposition(
        resA: pd.DataFrame,
        resB: pd.DataFrame,
        resAB: pd.DataFrame,
        name : str | None = "X",
        title: str | None = None,
        save_path: str | None = None,
        annotate_values: bool | None = False,
):
    if resA is None or resB is None or resAB is None:
        print(f"[{name}] Missing one of A/B/AB results, skipping plot.")
        return

    if title is None:
        title = f"[{name}] Granger (block) — partial R² vs horizon"

    fig, ax = plt.subplots(figsize=(6, 4))

    def plot_one(res_df, label, marker):
        ax.plot(res_df["horizon_min"], res_df["partial_r2_is"], marker=marker, label=label)
        if annotate_values:
            for x, y in zip(res_df["horizon_min"], res_df["partial_r2_is"]):
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom")

    plot_one(resA, "A: events + bolus flags", "o")
    plot_one(resB, "B: basal", "s")
    plot_one(resAB, "A+B: combined", "^")

    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Partial R² (in-sample)")
    ax.set_title(title)
    ax.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close()