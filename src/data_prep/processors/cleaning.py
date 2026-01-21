import pandas as pd
from typing import Any

from .interface import DataProcessor


class TypeAndValueCleaner(DataProcessor):
    """
    Processor responsible for:
    1. Coercing numeric columns to proper float/int types.
    2. Filling missing values (NaN) based on the default configuration.
    3. Clipping impulse columns (e.g. insulin) to be non-negative.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [TypeAndValueCleaner] Cleaning types, filling defaults and clipping...")

        # Estraiamo le info dal context
        # Nota: numeric_cols deve includere anche le colonne create dallo SchemaStandardizer se necessario
        # Ma di solito lo standardizer le crea già come float.
        numeric_cols = context.get('numeric_cols', [])
        defaults = context.get('defaults', {})
        impulse_cols = context.get('impulse_cols', [])

        for i, df in enumerate(data_list):

            # 1. Coerce Numeric Types
            # Forza le colonne che dovrebbero essere numeriche a diventarlo
            # (gestisce eventuali "0" stringa o errori di parsing)
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 2. Fill Defaults (Imputation Base)
            # Applica i valori di default definiti nel config (es. bolus=0.0 se NaN)
            for col, val in defaults.items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

            # Nota: Per le colonne create dallo SchemaStandardizer (che erano 0.0),
            # fillna non fa nulla perché sono già 0.0, quindi è coerente.

            # 3. Clip Impulse Columns
            # Assicura che insulina e carboidrati non siano mai negativi
            # (Utile se il dataset ha valori sporchi tipo -1)
            active_impulse_cols = [c for c in impulse_cols if c in df.columns]
            if active_impulse_cols:
                df[active_impulse_cols] = df[active_impulse_cols].clip(lower=0.0)

            data_list[i] = df

        return data_list