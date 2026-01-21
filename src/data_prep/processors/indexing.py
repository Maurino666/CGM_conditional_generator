import pandas as pd
from typing import Any

from .interface import DataProcessor


class TimeIndexer(DataProcessor):
    """
    Processor responsible for parsing the time column and setting it as the DataFrame index.
    It also sorts the data chronologically, which is crucial for time series tasks.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        time_col = context.get("time_col")

        if not time_col:
            raise ValueError("[TimeIndexer] 'time_col' not found in context.")

        print(f"   [TimeIndexer] Setting time index on column '{time_col}'...")

        for i, df in enumerate(data_list):
            # Se la colonna tempo non c'è, è un problema critico
            if time_col not in df.columns:
                # Controllo se per caso è già l'indice
                if df.index.name == time_col:
                    continue  # Già fatto, passo avanti
                raise ValueError(f"Time column '{time_col}' missing in subject {i}.")

            # 1. Convert to datetime (gestione UTC per coerenza)
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

            # 2. Drop rows where time parsing failed (NaT) - opzionale ma consigliato
            if df[time_col].isna().any():
                n_dropped = df[time_col].isna().sum()
                print(f"      Warning: Dropping {n_dropped} rows with invalid time in subject {i}")
                df = df.dropna(subset=[time_col])

            # 3. Set Index
            df = df.set_index(time_col)

            # 4. Sort Index (Cruciale per windowing e ffill)
            df = df.sort_index()

            data_list[i] = df

        return data_list