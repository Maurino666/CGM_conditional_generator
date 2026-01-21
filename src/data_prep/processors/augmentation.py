import pandas as pd
from pathlib import Path
from typing import Any

from .interface import DataProcessor
from ..utils import add_time_of_day_features, add_exponential_decay_feature


class BaseTimeEventAugmenter(DataProcessor):
    """
    Processor responsible for enriching the dataset with physiological and temporal dynamics.

    It transforms sparse, discrete events (like insulin boluses) into continuous
    physiological curves (IOB/COB) and encodes time cyclically. This "physics-informed"
    augmentation helps the model learn causal relationships more effectively.

    Transformations:
        Physiological Decay: Simulates Insulin On Board (IOB) and Carbs On Board (COB)
        using exponential decay functions on impulse columns.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Augments the dataset with derived dynamic features.

        Args:
            data_list: list of subject DataFrames.
            context: Pipeline context containing:
                     - 'impulse_cols': list of columns to apply decay to (e.g. bolus, carbs).
                     - 'time_steps': Timedelta representing the sampling frequency.
                     - 'logging_dir': Path for saving the augmentation report.

        Returns:
            list[pd.DataFrame]: DataFrames enriched with continuous dynamic columns.
        """
        print(f"   [PhysiologicalDynamicsAugmenter] Injecting physiological dynamics...")

        impulse_cols = context.get('impulse_cols', [])
        time_steps = context.get('time_steps')
        logging_dir = context.get('logging_dir')

        # Setup Logging
        log_file = None
        if logging_dir is not None:
            log_path = Path(logging_dir) / "augmented_features.txt"
            log_file = log_path.open("w", encoding="utf-8")

        try:
            for i, df in enumerate(data_list):
                added_cols_for_subject = []

                #  Physiological Decay (IOB/COB Simulation)
                # Transforms sparse impulses (spikes) into continuous curves representing
                # the active substance in the body over time.
                if time_steps is not None:
                    sampling_minutes = int(time_steps.total_seconds() / 60)

                    # Standard physiological half-life for rapid-acting insulin/carbs (~2 hours).
                    halflife_min = 120

                    for col in impulse_cols:
                        # Safety Check: Augment only if the column exists in this dataset.
                        if col in df.columns:
                            df, new_col_name = add_exponential_decay_feature(
                                df,
                                col=col,
                                sampling_minutes=sampling_minutes,
                                halflife_min=halflife_min,
                                past_only=True,
                            )
                            added_cols_for_subject.append(new_col_name)

                # Update the DataFrame in the list
                data_list[i] = df

                # Log added columns
                if log_file is not None:
                    print(f"Subject {i + 1}:", file=log_file)
                    for c in added_cols_for_subject:
                        print(f"  - {c}", file=log_file)
                    print("", file=log_file)

        finally:
            if log_file is not None:
                log_file.close()

        return data_list