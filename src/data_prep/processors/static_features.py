from typing import  Any
from pathlib import Path
import yaml
import pandas as pd
from .interface import DataProcessor


class StaticFeaturesProcessor(DataProcessor):
    """
    Processor that injects raw static features (demographic/clinical) into DataFrames.

    It reads from a simple YAML metadata file containing subject data and applies 
    constant values (broadcasting) to all time steps for each subject.

    NOTE: This processor does NOT perform normalization or encoding. 
    It injects values exactly as they appear in the YAML file.
    Downstream processors are responsible for scaling or type conversion.
    """

    def __init__(self, metadata_path: Path | None = None):
        """
        Args:
            metadata_path: Path to the .yaml metadata file containing subject info.
                           If None or the file does not exist, the processor will be inactive.
        """
        self.metadata = self._load_metadata(metadata_path)

    def _load_metadata(self, path: Path | None = None) -> dict | None:
        """
        Safely loads the YAML file. Returns None if path is invalid or load fails.
        """
        if path is None or not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[StaticFeaturesProcessor] Warning: Failed to load metadata from {path}: {e}")
            return None

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Enrich each dataframe in the list with raw static features based on the subject ID.
        """
        # If no metadata is loaded, return the data list unchanged (Pass-through)
        if self.metadata is None:
            return data_list

        print(f"   [StaticFeaturesProcessor] Injecting raw static features for {len(data_list)} subjects...")

        # In this simplified version, we expect the YAML to directly contain subject IDs as keys.
        # Structure: {'559': {'age': 45, 'gender': 1.0}, ...}
        subjects_data = self.metadata

        # Attempt to determine subject IDs from context
        subject_ids = context.get("subject_ids", [str(i) for i in range(len(data_list))])

        for i, df in enumerate(data_list):
            if i >= len(subject_ids):
                break

            sid = str(subject_ids[i])
            subj_info = subjects_data.get(sid)

            # If no info exists for this subject, skip.
            # (Alternatively, one could inject NaNs, but skipping is safer for sparse metadata)
            if subj_info is None:
                continue

            # Apply raw features
            data_list[i] = self._enrich_dataframe(df, subj_info)

        return data_list

    def _enrich_dataframe(self, df: pd.DataFrame, subj_info: dict) -> pd.DataFrame:
        """
        Applies raw values broadcasting to the single DataFrame.
        """
        # Iterate over all key-value pairs in the subject's dictionary
        for feat_name, raw_val in subj_info.items():

            # Define column names
            col_name = f"static_{feat_name}"

            try:
                # --- Broadcasting ---
                # Assign the raw value directly. 
                # Pandas handles broadcasting the scalar to the entire index.
                df[col_name] = raw_val

            except Exception as e:
                print(f"[StaticFeaturesProcessor] Error setting {feat_name}: {e}")
                continue

        return df