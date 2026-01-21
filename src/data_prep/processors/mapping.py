import pandas as pd
from typing import Any

from .interface import DataProcessor


class ColumnMapper(DataProcessor):
    """
    Processor responsible for renaming columns from the raw dataset format
    to the internal standard format defined in the configuration.

    It performs:
    1. Whitespace stripping from column names.
    2. Renaming based on the 'col_mapping' dictionary.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        mapping = context.get("col_mapping", {})

        print(f"   [ColumnMapper] Renaming columns using provided mapping...")

        # Invertiamo il mapping per usarlo nel rename di pandas
        # Il config è solitamente: standard_name -> raw_name (es. timestamp: EventDateTime)
        # Pandas vuole: raw_name -> standard_name
        inverse_mapping = {raw: standard for standard, raw in mapping.items()}

        for i, df in enumerate(data_list):
            # 1. Strip whitespace from column names (common CSV issue)
            df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

            # 2. Apply mapping
            # Se una colonna non è nel mapping, rimane col suo nome originale
            # (verrà scartata dopo dallo SchemaStandardizer)
            df = df.rename(columns=inverse_mapping)

            data_list[i] = df

        return data_list