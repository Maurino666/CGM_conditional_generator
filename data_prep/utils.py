import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt

import yaml


def load_dataset_config(config_path: Path) -> dict[str, Any]:
    """
    Load a dataset configuration YAML file into a Python dictionary.

    Example:
        config = load_dataset_config("azt1d", Path("configs"))

    :param config_path:  Config file path.
    :return:             Parsed configuration as a dictionary.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Open the YAML file and parse it into a Python dict
    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    # Minimal sanity checks (you can add more if you want)
    if "dataset" not in config:
        raise ValueError(f"Config file '{config_path}' has no top-level 'dataset' section.")

    if "schema" not in config:
        raise ValueError(f"Config file '{config_path}' has no top-level 'schema' section.")

    return config



# Loading dataset

def load_dataset(
        data_root: Path,
        sep: str | None = ","
) -> list[pd.DataFrame]:
    """
    Load dataset from the specified root directory.

    Every CSV file found under the root directory (recursively) is treated
    as a separate patient file.

    :param data_root: Path to the root directory containing patient CSV files.
    :param sep: Separator for CSV files.
    :return:         List of DataFrames, one per patient CSV.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {data_root}")

    # Collect all CSV files under root (recursive search)
    csv_files = sorted(
        p for p in data_root.rglob("*.csv") if p.is_file()
    )
    print(f"Found {len(csv_files)} patient CSV files under root (recursive search).")

    all_data = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, sep=sep)
        except Exception as e:
            # Skip files that cannot be read
            print(f"Skipping {csv_path} due to read error: {e}")
            continue

        all_data.append(df)
        print(f"Loaded {csv_path} with shape {df.shape}")

    return all_data

from pathlib import Path
from typing import List
import sys

import pandas as pd


def print_df_summary(
        all_data: List[pd.DataFrame],
        logging_path: Path | None = None
) -> None:
    """
    Print a summary for each DataFrame in all_data.
    If logging_path is provided, write the summary to that file instead of stdout.
    """

    # Decide where to write: stdout or a log file
    if logging_path is None:
        out = sys.stdout
        close_out = False
    else:
        out = logging_path.open("w", encoding="utf-8")
        close_out = True

    try:
        for i, df in enumerate(all_data):
            print(
                f"\n📊 DataFrame summary Subject {i + 1}: ({len(df):,} rows, {len(df.columns)} columns)",
                file=out,
            )
            print("=" * 100, file=out)

            for col in df.columns:
                s = df[col]
                dtype = str(s.dtype)
                nunique = s.nunique(dropna=True)
                missing = s.isna().mean() * 100

                print(f"\n🔹 {col}  [{dtype}]", file=out)
                print(f"   • unique values: {nunique}", file=out)
                print(f"   • missing values: {missing:.1f}%", file=out)

                if pd.api.types.is_numeric_dtype(s):
                    desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
                    print(
                        f"   • min={desc['min']:.2f}, median={desc['50%']:.2f}, max={desc['max']:.2f}",
                        file=out,
                    )
                    mode_vals = s.value_counts(dropna=True).head(5)
                    print("   • most frequent values:", file=out)
                    for val, count in mode_vals.items():
                        # Here we assume val is numeric; if not, you may want to cast to str
                        print(f"       {val:.2f} → {count}", file=out)
                else:
                    mode_vals = s.value_counts(dropna=True).head(5)
                    if mode_vals.empty:
                        print("   • no non-null values", file=out)
                    else:
                        print("   • most frequent values:", file=out)
                        for val, count in mode_vals.items():
                            print(f"       {val} → {count}", file=out)

                print("-" * 100, file=out)
    finally:
        # Close the file only if we opened it
        if close_out:
            out.close()

def print_duplicate_counts(
        all_data: List[pd.DataFrame],
        logging_path: Path | None = None
):
    # Decide where to write: stdout or a log file
    if logging_path is None:
        out = sys.stdout
        close_out = False
    else:
        out = logging_path.open("w", encoding="utf-8")
        close_out = True

    try:
        for i, df in enumerate(all_data, start=1):
            subject = f"Subject {i:02d}"

            if df is None or df.empty:
                print(f"[{subject}] empty DataFrame", file=out)
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"[{subject}] index is not a DatetimeIndex", file=out)
                continue

            # Find duplicated rows by timestamp (in the index)
            dup_mask = df.index.duplicated(keep=False)
            n_dup_rows = int(dup_mask.sum())
            if n_dup_rows == 0:
                print(f"[{subject}] no duplicate timestamps", file=out)
                continue

            # How many distinct duplicated timestamps?
            n_dup_timestamps = df.index[dup_mask].nunique()

            print(
                f"[{subject}] {n_dup_rows} duplicate rows across "
                f"{n_dup_timestamps} duplicated timestamps",
                file=out
            )

            # Show first 3 duplicated timestamps with their counts
            vc = df.index.value_counts().sort_index()
            dups = vc[vc > 1].head(3)
            for ts, cnt in dups.items():
                print(f"   - {ts}  (count = {cnt})", file=out)
    finally:
        # Close the file only if we opened it
        if close_out:
            out.close()


def clean_duplicates(
  all_data:  list[pd.DataFrame]
) -> list[pd.DataFrame]:

    """
    Clean duplicate timestamps in each DataFrame by keeping the first occurrence.
    :param all_data: List of DataFrames to be cleaned.
    :return: List of cleaned DataFrames.
    """

    all_data_nodup = []
    for i in range(len(all_data)):
        df = all_data[i]
        if df is None or df.empty:
            all_data_nodup.append(df)
            print(f"[Subject {i+1}] empty")
            continue

            # Drop duplicates in the index, keeping the first occurrence
        df_nodup = df[~df.index.duplicated(keep="first")]

        # Store cleaned DataFrame
        all_data_nodup.append(df_nodup)

        n_removed = len(df) - len(df_nodup)
        if n_removed > 0:
            print(f"[Subject {i+1}] removed {n_removed} duplicate rows")
        else:
            print(f"[Subject {i+1}] no duplicates removed")

    return all_data_nodup

def fill_data(
        all_data : list[pd.DataFrame],
        expected : pd.Timedelta,
        max_gap : pd.Timedelta,
        target_col : str,

        defaults : dict = None,
        logging_path : Path | None = None
) -> list[pd.DataFrame]:

    """
    Fill missing data in each DataFrame by resampling and interpolating the target column.
    Missing values in other columns are filled with specified default values.
    Target column is interpolated only for gaps smaller than max_gap.
    :param all_data: data to be filled
    :param expected: expected sampling interval
    :param max_gap: maximum gap to fill
    :param target_col: target column to interpolate
    :param defaults: dictionary of default values for other columns
    :param logging_path: path where effect graphs will be saved
    :return: data with filled values
    """
    limit_steps = int(max_gap / expected)
    all_data_filled = []

    for i in range(len(all_data)):

        df = all_data[i]
        if df is None or df.empty:
            all_data_filled.append(df)
            print(f"[Subject {i+1}] empty")
            continue

        origin_time = df.index.min()
        df_full = df.resample(expected, origin=origin_time).last()

        target = pd.to_numeric(df_full[target_col], errors='coerce')
        target_filled = target.interpolate(
            method='time',
            limit= limit_steps,  # <= 4 points -> 20 minutes
            limit_area='inside'  # no tail extrapolation
        )

        print(f"[Subject {i+1}] filled {len(df_full)} rows")
        df_full[target_col] = target_filled

        mask_inferred = target.isna() & target_filled.notna()

        if defaults is not None:
            for col, val in defaults.items():
                if col in df_full:
                    df_full.loc[mask_inferred, col] = val
                else:
                    print(f"[Subject {i+1}] missing {col}")
        else:
            print(f"[Subject {i+1}] no defaults provided")

        all_data_filled.append(df_full)

    if logging_path is not None:
        plot_gaps(all_data, all_data_filled, target_col, expected, save_dir=logging_path)
    return all_data_filled

def plot_gaps(
        all_data : list[pd.DataFrame],
        all_data_filled : list[pd.DataFrame],
        target_col : str,
        expected : pd.Timedelta,
        show_plot: bool = False,
        save_dir : Path | None = None,
):
    """
    Plots filled and unfilled gaps in the timestamps comparing lists of dataframes
    :param all_data: list of unfilled dataframes
    :param all_data_filled: list of filled dataframes
    :param target_col: target column
    :param expected: expected sampling interval
    :param show_plot: weather to show the resulting plots
    :param save_dir: path of the directory where files will be saved
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(all_data)):
        p_df_after = all_data_filled[i]  # AFTER interpolation (on exact grid)
        p_df_before = all_data[i].sort_index()  # BEFORE (raw)

        # Snap raw to 5-min bins and deduplicate, then align exactly to AFTER index
        # 1. Trova l'origine esatta usata nel riempimento
        # (Non è necessario ricalcolarla, è già nel DataFrame finale se lo ricrei)
        origin_time = p_df_before.index.min()

        # 2. Allinea il DataFrame RAW (originale) alla griglia OFFSET PERFETTA
        #    usando la stessa logica di fill_data (resample.last).
        p_df_before_on_grid = p_df_before.resample(expected, origin=origin_time).last()

        # 3. Riallinea l'indice: Se p_df_before_on_grid avesse una lunghezza diversa,
        #    la forziamo ad allinearsi all'indice perfetto di p_df_after (che è il riferimento)
        p_df_before_on_grid = p_df_before_on_grid.reindex(p_df_after.index)

        # Masks on the same grid
        mask_inferred = p_df_before_on_grid[target_col].isna() & p_df_after[target_col].notna()
        mask_still_na = p_df_before_on_grid[target_col].isna() & p_df_after[target_col].isna()

        # Helper to group contiguous timestamps into segments
        def to_segments(ts_index, base_idx):
            if len(ts_index) == 0:
                return []
            sampling = base_idx.to_series().diff().median()
            brk = ts_index.to_series().diff() > sampling * 1.1
            grp = brk.cumsum()
            grouped = ts_index.to_series().groupby(grp)
            return [(g.min(), g.max()) for _, g in grouped]

        # Use the AFTER index as base (regular grid)
        idx = p_df_after.index

        inferred_times = idx[mask_inferred]
        persist_times = idx[mask_still_na]  # <-- fixed name

        segments_inferred = to_segments(inferred_times, idx)
        segments_persist = to_segments(persist_times, idx)

        # (Optional) structural gaps: with a regular grid they are empty;
        # if you still want orange bars, compute gaps from the ORIGINAL raw index BEFORE rounding.
        gaps = []  # leave empty to avoid confusion on the regular grid

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 2.2))
        ax.hlines(y=0, xmin=idx.min(), xmax=idx.max(), color="black", linewidth=5, alpha=0.35)

        # Structural gaps (orange) - only if you computed them above
        for ts, delta in gaps:
            ax.hlines(y=0, xmin=ts - delta, xmax=ts, color="orange", linewidth=10, label="Index gaps > threshold")
            break  # ensures single legend entry

        # Still NaN (red)
        first = True
        for start, end in segments_persist:
            ax.hlines(y=0, xmin=start, xmax=end, color="red", linewidth=10,
                      label="Still NaN (before & after)" if first else "")
            first = False

        # Inferred (blue)
        first = True
        for start, end in segments_inferred:
            ax.hlines(y=0, xmin=start, xmax=end, color="blue", linewidth=10,
                      label="Inferred" if first else "")
            first = False

        ax.set_yticks([])
        ax.set_title(f"{i + 1} — CGM inferred (blue), still-NaN (red)")
        ax.set_xlabel("Time")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")
        plt.tight_layout()

        if show_plot:
            plt.show()

        if save_dir is not None:
            plt.savefig(save_dir / f"subject_{i + 1}.png")
            print(f"Saved plot in {save_dir / f"subject_{i + 1}.png"}")

        plt.close(fig)


def add_target_lags(
    df: pd.DataFrame,
    target_col: str,
    lag_minutes: tuple[int, ...],
    sampling_minutes: int = 5,
    enforce_multiple: bool = True,
) -> pd.DataFrame:
    """Return df with columns like `target_lag_30m` computed via shift; no in-place ops."""
    new_cols = {}
    for m in lag_minutes:
        if enforce_multiple and (m % sampling_minutes != 0):
            raise ValueError(f"Lag {m} min is not a multiple of sampling interval {sampling_minutes} min.")
        steps = int(round(m / sampling_minutes))
        new_cols[f"{target_col}_lag_{m}m"] = df[target_col].shift(steps)
    return df.assign(**new_cols)


def add_time_of_day_features(
    df: pd.DataFrame,
    include_12h: bool = False,
    datetime_col: str | None = None,
) -> tuple[pd.DataFrame, list]:
    """
    Add sinusoidal time-of-day features to a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data, with either a DatetimeIndex or a datetime column.
    datetime_col : str | None
        Datetime column to use; if None, use the index.
    include_12h : bool
        If True, also add 12-hour harmonics.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with added time-of-day columns.
    added_cols : list
        Names of the new feature columns.
    """
    if datetime_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex if datetime_col is None.")
        t = pd.Series(df.index, index=df.index)
    else:
        t = pd.to_datetime(df[datetime_col])
        if t.isna().any():
            raise ValueError("datetime_col contains non-parsable or missing datetimes.")
    hour = t.dt.hour + t.dt.minute / 60.0

    new_cols = {
        "tod_sin_24h": np.sin(2 * np.pi * hour / 24.0),
        "tod_cos_24h": np.cos(2 * np.pi * hour / 24.0),
    }
    if include_12h:
        new_cols.update({
            "tod_sin_12h": np.sin(2 * np.pi * hour / 12.0),
            "tod_cos_12h": np.cos(2 * np.pi * hour / 12.0),
        })
    return df.assign(**new_cols), list(new_cols.keys())

def add_event_present_indicator(
    df: pd.DataFrame,
    col: str,
    threshold: float = 0.0,
    strictly_greater: bool = True,
    out_name: str | None = None,
    dtype: str = "int8",
) -> pd.DataFrame:
    """
    Create a binary indicator at time t for a sparse, nonnegative event column.
    - If strictly_greater=True: present = 1 when value > threshold
      else present = 1 when value >= threshold
    - NaN values are treated as 0 (i.e., not present).

    Parameters
    ----------
    df : input DataFrame
    col : event column name (e.g., "CarbSize", "TotalBolusInsulinDelivered")
    threshold : decision threshold (default 0.0)
    strictly_greater : use '>' (True) or '>=' (False)
    out_name : optional output column name; defaults to f"{col}_present"
    dtype : output integer dtype (e.g., "int8")

    Returns
    -------
    DataFrame with one new column: {col}_present (or `out_name`).
    """
    x = df[col].fillna(0.0)
    mask = (x > threshold) if strictly_greater else (x >= threshold)
    present = mask.astype(dtype)
    name = out_name or f"{col}_present"
    return df.assign(**{name: present})

def add_exponential_decay_feature(
    df: pd.DataFrame,
    col: str,
    time_col: str | None = None,
    sampling_minutes: int | None = 5,
    halflife_min: float = 60.0,
    use_magnitude: bool = True,
    past_only: bool = True,
    out_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Add an exponential-decay memory feature for a sparse event series `col`.
    Recurrence: r_t = alpha_t * r_{t-1} + x_{t-1}  (past-only to avoid leakage)

    Parameters
    ----------
    df : input DataFrame
    col : event column name (nonnegative). If use_magnitude=False, we use presence (0/1).
    time_col : optional datetime column. If None, uses DateTimeIndex.
    sampling_minutes : if time_col is None and sampling is regular, provide step in minutes.
                       Ignored when time_col is given (we compute variable Δt).
    halflife_min : half-life in minutes (time to halve the residual memory)
    use_magnitude : True => use raw values; False => use presence (0/1)
    past_only : if True, uses x_{t-1}; if False, uses x_{t} (beware of leakage in forecasting)
    out_name : optional output column name

    Returns
    -------
    DataFrame with one new column: {col}_expdecay_{halflife_min}m  (or `out_name`).
    column name: name of the added column
    """
    # Input signal (fill NaN with 0; cast presence if requested)
    x = df[col].fillna(0.0)
    if not use_magnitude:
        x = (x > 0).astype(float)

    # Resolve time deltas in minutes
    if time_col is not None:
        t = pd.to_datetime(df[time_col])
        if t.isna().any():
            raise ValueError("time_col contains non-parsable or missing datetimes.")
        dt_min = t.diff().dt.total_seconds().div(60.0).bfill()  # first step ~ next
    else:
        if not isinstance(df.index, pd.DatetimeIndex) and sampling_minutes is None:
            raise ValueError("Provide time_col or a sampling_minutes for regular grids.")
        if isinstance(df.index, pd.DatetimeIndex):
            dt_min = df.index.to_series().diff().dt.total_seconds().div(60.0).bfill()
        else:
            dt_min = pd.Series(float(sampling_minutes), index=df.index)

    # Compute per-step alpha from half-life: alpha = 0.5 ** (Δt / HL)
    alpha = np.power(0.5, dt_min.astype(float) / float(halflife_min)).to_numpy()

    # Choose which x to use (past-only avoids leakage)
    x_arr = x.shift(1).fillna(0.0).to_numpy() if past_only else x.to_numpy()

    # Run the leaky integrator
    r = np.empty(len(df), dtype=float)
    acc = 0.0
    for i in range(len(df)):
        acc = alpha[i] * acc + x_arr[i]
        r[i] = acc

    name = out_name or f"{col}_expdecay_{int(round(halflife_min))}m"
    return df.assign(**{name: pd.Series(r, index=df.index)}), name


def encode_bolus_type_semantic(
    df: pd.DataFrame,
    col: str,
    out_prefix: str = "bt_",
) -> pd.DataFrame:
    """
    Semantic encoding for a categorical 'BolusType' column.

    Outputs (added columns):
      - {p}is_standard  (int8)  : 1 if 'standard' (or inferred as standard when not extended/dual and marked quick/auto)
      - {p}is_extended  (int8)  : 1 if 'extended' present
      - {p}is_dual      (int8)  : 1 if 'dual'/'combo'/'multiwave' present
      - {p}is_correction(int8)  : 1 if 'correction' present
      - {p}is_auto      (int8)  : 1 if 'auto'/'automatic' present (e.g., 'automatic bolus/correction')
      - {p}is_quick     (int8)  : 1 if 'quick' present
      - {p}is_ble       (int8)  : 1 if 'ble' present
      - {p}ext_fraction (float) : in [0, 1], from patterns like 'extended 50.00%/15.00'; NaN otherwise
      - {p}ext_duration (float) : numeric duration from same pattern; NaN otherwise

    Assumptions:
      - 'col' is present; non-string values are cast to string.
      - Labels may include parameters in-text (e.g., 'extended 50%/15').
      - 'none' means no bolus: all flags 0; parameters NaN.

    Returns a new DataFrame with added columns (no in-place operations).
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Normalize to string (pandas StringDtype), lowercase, trim, compress spaces
    s = df[col].astype("string")
    s_norm = (
        s.str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.lower()
    )

    # Basic presence flags
    is_none = s_norm.eq("none")
    is_extended = s_norm.str.contains(r"\bextended\b", na=False)
    is_dual = s_norm.str.contains(r"\bdual\b|\bcombo\b|\bmultiwave\b", na=False)
    is_correction = s_norm.str.contains(r"\bcorrection\b", na=False)
    is_auto = s_norm.str.contains(r"\bauto(?:matic)?\b", na=False)
    is_quick = s_norm.str.contains(r"\bquick\b", na=False)
    is_ble = s_norm.str.contains(r"\bble\b", na=False)

    # Standard: explicit 'standard' OR inferred when not extended/dual but marked quick/auto/ble-standard
    explicit_standard = s_norm.str.contains(r"\bstandard\b|\bnormal\b", na=False)
    inferred_standard = (~is_extended & ~is_dual) & (is_quick | is_auto | is_ble)
    is_standard = (explicit_standard | inferred_standard)

    # Extended parameters: extract "<percent>%/<duration>" anywhere in the string
    # Example: "extended/correction 50.00%/15.00"
    # Group 1 = percent, Group 2 = duration
    # Apply only where 'extended' is present to avoid accidental matches
    ext_str = s_norm.where(is_extended)
    ext_params = ext_str.str.extract(r"(\d+(?:\.\d+)?)\s*%\s*/\s*([0-9]+(?:\.[0-9]+)?)")

    # Convert to numeric; percent -> fraction in [0,1]
    ext_fraction = pd.to_numeric(ext_params[0], errors="coerce") / 100.0
    if ext_fraction.notna().any():
        ext_fraction = ext_fraction.clip(lower=0.0, upper=1.0)
    ext_duration = pd.to_numeric(ext_params[1], errors="coerce")

    # Rows with 'none' => all flags 0, parameters NaN
    zero = np.int8(0)
    one  = np.int8(1)
    out = df.assign(
        **{
            f"{out_prefix}is_standard": is_standard.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_extended": is_extended.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_dual":     is_dual.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_correction": is_correction.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_auto":     is_auto.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_quick":    is_quick.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}is_ble":      is_ble.fillna(False).astype("int8").where(~is_none, zero),
            f"{out_prefix}ext_fraction": ext_fraction.where(~is_none, np.nan),
            f"{out_prefix}ext_duration": ext_duration.where(~is_none, np.nan),
        }
    )

    return out


def build_sliding_windows(
    all_data: list[pd.DataFrame],
    feature_cols: list[str],
    seq_len: int,
    step: int,
    max_missing_ratio: float = 0.0,
) -> np.ndarray:
    """
    Build contiguous sliding windows from a list of patient DataFrames.

    Each DataFrame is assumed to contain a time series for a single subject,
    already cleaned and aligned on a regular time grid (e.g. every 5 minutes).

    For each subject, the function extracts windows of length `seq_len` by
    moving a sliding window with stride `step` over the rows. Windows that
    contain too many missing values (above `max_missing_ratio`) are discarded.

    Parameters
    ----------
    all_data : list of pd.DataFrame
        List of patient DataFrames.
    feature_cols : sequence of str
        Column names to use as features in the windows.
    seq_len : int
        Length of each window (number of time steps).
    step : int
        Sliding window step (number of rows between the start of consecutive windows).
    max_missing_ratio : float, optional
        Maximum allowed fraction of NaN values inside a window (between 0 and 1).
        Windows with a higher missing ratio are discarded. Default is 0.0 (no NaNs allowed).

    Returns
    -------
    np.ndarray
        3D array of shape (num_windows, seq_len, num_features), dtype float32.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if not 0.0 <= max_missing_ratio <= 1.0:
        raise ValueError("max_missing_ratio must be between 0.0 and 1.0")

    windows: list[np.ndarray] = []

    for df_idx, df in enumerate(all_data):
        if df.empty:
            continue

        if df.index is not None:
            df = df.sort_index()

        # Check that all required feature columns are present
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise KeyError(
                f"DataFrame {df_idx} is missing required feature columns: {missing_features}"
            )

        # Restrict to feature columns
        sub = df[list(feature_cols)].copy()
        num_rows = len(sub)

        if num_rows < seq_len:
            # Not enough data points for a single window
            continue

        # Slide over the DataFrame rows
        start = 0
        while start + seq_len <= num_rows:
            window = sub.iloc[start : start + seq_len]

            # Compute missing ratio across all features and time steps
            missing_ratio = window.isna().mean().mean()

            if missing_ratio <= max_missing_ratio:
                # Convert to numpy array, cast to float32 for efficiency
                values = window.to_numpy(dtype=np.float32)
                # values shape: (seq_len, num_features)
                windows.append(values)

            start += step

    if not windows:
        # No valid windows were found
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)

    # Stack all windows into a single 3D array
    X = np.stack(windows, axis=0)  # shape: (num_windows, seq_len, num_features)
    return X

def build_sliding_windows_conditional(
    all_data: list[pd.DataFrame],
    seq_len: int,
    step: int,
    target_col: str,
    cond_cols: list[str],
    max_missing_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows and split them into target and conditioning parts.

    Parameters
    ----------
    all_data : list[pd.DataFrame]
        Sequence of subject DataFrames.
    seq_len : int
        Window length (number of time steps).
    step : int
        Stride between consecutive windows.
    target_col : str
        Name of the target column (first feature in the windows).
    cond_cols : list[str]
        Names of conditioning columns to include in the windows.
    max_missing_ratio: float, Optional
        Max ratio of missing values in window.

    Returns
    -------
    X_target : np.ndarray
        Target windows of shape (num_windows, seq_len, 1).
    X_cond : np.ndarray
        Conditioning windows of shape (num_windows, seq_len, num_cond_features).
    """
    # Ensure target is not duplicated in cond_cols
    cond_cols_clean = [c for c in cond_cols if c != target_col]

    # Complete feature list: target first, then conditioning
    all_cols: list[str] = [target_col] + cond_cols_clean

    # Sanity check: required columns must exist in all DataFrames
    for i, df in enumerate(all_data):
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"DataFrame {i} is missing required columns for sliding windows: {missing}"
            )

    # Use the existing generic sliding-window builder
    X_full = build_sliding_windows(
        all_data = all_data,
        seq_len=seq_len,
        step=step,
        feature_cols=all_cols,
        max_missing_ratio=max_missing_ratio,
    )
    # X_full shape: (num_windows, seq_len, 1 + len(cond_cols_clean))

    # Split into target (first feature) and conditioning (remaining features)
    X_target = X_full[:, :, :1]        # (N, seq_len, 1)
    X_cond = X_full[:, :, 1:]          # (N, seq_len, num_cond_features)

    return X_target, X_cond

