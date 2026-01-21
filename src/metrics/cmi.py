import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from pathlib import Path

def _standardize(*arrays, jitter: float = 1e-6, rng: np.random.RandomState | None = None):
    """
    Standardizza ogni array per colonna e aggiunge un piccolo jitter
    (per evitare legami esatti e distanze nulle).
    """
    out = []
    for X in arrays:
        X = np.asarray(X)
        if X.size == 0:
            out.append(X)
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if jitter and Xs.size:
            eps = (np.std(Xs, axis=0, keepdims=True) + 1e-12) * jitter
            noise = (rng or np.random).normal(0.0, 1.0, size=Xs.shape) * eps
            Xs = Xs + noise
        out.append(Xs)
    return out


# KSG Conditional Mutual Information
def cmi_ksg(X, Y, Z, k: int = 10, metric: str = 'chebyshev') -> float:
    """
    Stima KSG/Frenzel–Pompe di I(X;Y | Z) in bit.
    - ε(i): distanza al k-esimo vicino nel joint (X, Y, Z) con metrica L∞ (Chebyshev).
    - Nei sottospazi contiamo i vicini con distanza < ε(i), escludendo il punto stesso.
    - Formula con ψ(k) e senza il termine ψ(n).
    """
    def _as_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    X = _as_2d(X); Y = _as_2d(Y); Z = _as_2d(Z)
    n = X.shape[0]
    if n == 0:
        return np.nan

    XZ  = np.concatenate([X, Z], axis=1) if Z.size else X
    YZ  = np.concatenate([Y, Z], axis=1) if Z.size else Y
    XYZ = np.concatenate([X, Y, Z], axis=1) if Z.size else np.concatenate([X, Y], axis=1)

    # k-esimo vicino nel joint
    nn_joint = NearestNeighbors(metric=metric, n_neighbors=k+1).fit(XYZ)
    dists, _ = nn_joint.kneighbors(XYZ, n_neighbors=k+1)
    eps = dists[:, k]
    tol = 1e-12

    def count_lt_eps(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return np.zeros(n, dtype=int)
        nn = NearestNeighbors(metric=metric).fit(mat)
        out = np.empty(n, dtype=int)
        for i in range(n):
            r = max(eps[i] - tol, 0.0)
            if r <= 0.0:
                out[i] = 0
            else:
                ind = nn.radius_neighbors(mat[i:i+1], radius=r, return_distance=False)[0]
                out[i] = len(ind) - 1   # escludi il punto stesso
        return out

    nxz = count_lt_eps(XZ)
    nyz = count_lt_eps(YZ)
    nz  = count_lt_eps(Z) if Z.size else np.zeros(n, dtype=int)

    # KSG-CMI in nats → bit  (NESSUN +ψ(n)!)
    val_nats = digamma(k) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
    return float(val_nats / np.log(2.0))

# ==============================
# Decomposizione CMI su orizzonti
# ==============================
def cmi_decomposition_over_horizons(
    df: pd.DataFrame,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    k: int = 10,
    # metric tenuto per compatibilità, ma internamente usiamo sempre Chebyshev
    metric: str = "chebyshev",
    jitter: float = 1e-6,
    random_state: int = 0,
    add_time_of_day: bool = True,
    clip_at_zero: bool = True,
) -> pd.DataFrame:
    """
    Per ogni orizzonte h:
      Y = target spostato di -h (in step da freq_min)
      Stima KSG di:
        I_total  = I([A,B]; Y | base)
        unique_A = I(A; Y | base, B)
        unique_B = I(B; Y | base, A)
      shared_raw = I_total - unique_A - unique_B
      synergy = max(shared_raw, 0), overlap = max(-shared_raw, 0) se clip_at_zero=True
    """
    rng = np.random.RandomState(random_state)
    rows = []

    needed = set(base_cols)
    if add_time_of_day:
        needed |= set([c for c in ("tod_sin_24h", "tod_cos_24h") if c in df.columns])
    needed |= set(features_A) | set(features_B)
    needed = list(needed)

    for h in horizons_min:
        step = int(round(h / float(freq_min)))
        if step <= 0:
            continue

        y = df[target_col].shift(-step).rename("_y_")
        idx = df[needed].join(y).dropna().index
        if len(idx) < 100:
            rows.append({
                "horizon_min": h, "n": len(idx),
                "I_total": np.nan, "unique_A": np.nan, "unique_B": np.nan,
                "shared_raw": np.nan, "synergy": np.nan, "overlap": np.nan,
                "k": k, "metric": "chebyshev", "unit": "bits",
            })
            continue

        # Design
        Base = df.loc[idx, base_cols].copy()
        if add_time_of_day:
            for c in ("tod_sin_24h", "tod_cos_24h"):
                if c in df.columns:
                    Base[c] = df.loc[idx, c]
        A = df.loc[idx, features_A].copy()
        B = df.loc[idx, features_B].copy()
        Y = y.loc[idx].values.reshape(-1, 1)

        # Standardizzazione + jitter
        Base_s, A_s, B_s, Y_s = _standardize(Base.values, A.values, B.values, Y,
                                              jitter=jitter, rng=rng)

        # CMI (Chebyshev fissata dentro cmi_ksg)
        I_total  = cmi_ksg(np.concatenate([A_s, B_s], axis=1), Y_s, Base_s, k=k)
        unique_A = cmi_ksg(A_s, Y_s, np.concatenate([Base_s, B_s], axis=1), k=k)
        unique_B = cmi_ksg(B_s, Y_s, np.concatenate([Base_s, A_s], axis=1), k=k)

        shared_raw = I_total - unique_A - unique_B
        synergy = max(0.0, shared_raw) if clip_at_zero else shared_raw
        overlap = max(0.0, -shared_raw) if clip_at_zero else 0.0

        rows.append({
            "horizon_min": h, "n": len(idx),
            "I_total": I_total,
            "unique_A": unique_A,
            "unique_B": unique_B,
            "shared_raw": shared_raw,
            "synergy": synergy,
            "overlap": overlap,
            "k": k, "metric": "chebyshev", "unit": "bits",
        })

    return pd.DataFrame(rows)


# ==============================
# Plot
# ==============================
def plot_cmi_decomposition(
        df_cmi: pd.DataFrame,
        name: str | None = "X",
        title: str = None,
        save_path : Path | None = None,
):
    if title is None:
        title =f"[{name}] CMI Decomposition results"

    x = df_cmi["horizon_min"].values

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, df_cmi["I_total"],  "-o", label="I_total = I(Y; A,B | base)")
    plt.plot(x, df_cmi["unique_A"], "-s", label="unique A | B")
    plt.plot(x, df_cmi["unique_B"], "-^", label="unique B | A")

    if "synergy" in df_cmi.columns:
        plt.plot(x, df_cmi["synergy"], "--", label="synergy (≥0)")
    if "overlap" in df_cmi.columns:
        plt.plot(x, df_cmi["overlap"], ":", label="overlap (≥0)")

    plt.xlabel("Orizzonte (min)")
    plt.ylabel("Informazione condizionale [bit]")
    plt.title(title)
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path / f"{name}.png")
        print(f"Saved plot in {save_path / f"{name}.png"}")

    plt.close()
