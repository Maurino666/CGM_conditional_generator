from pathlib import Path
from typing import Any

import pandas as pd

from .interface import DataProcessor
from .. import add_time_of_day_features


class TodProcessor(DataProcessor):

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        logging_dir = context.get('logging_dir')

        # Setup Logging
        log_file = None
        if logging_dir is not None:
            log_path = Path(logging_dir) / "tod_features.txt"
            log_file = log_path.open("w", encoding="utf-8")

        new_data_list = []
        try:
            for i, df in enumerate(data_list):
                #  Time of Day (Cyclical Encoding)
                # Transforms linear time (00:00 -> 23:59) into cyclical features (sin, cos)
                # ensuring 23:59 is mathematically close to 00:00.
                df, tod_cols = add_time_of_day_features(df)
                new_data_list.append(df)

                # Log added columns
                if log_file is not None:
                    print(f"Subject {i + 1}:", file=log_file)
                    for c in tod_cols:
                        print(f"  - {c}", file=log_file)
                    print("", file=log_file)
        finally:
            if log_file is not None:
                log_file.close()

        return new_data_list
