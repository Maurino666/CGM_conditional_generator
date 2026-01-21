from pathlib import Path
from .base_dataset import BaseDataset


class OhioT1DMDataset(BaseDataset):
    """
    Dataset class for OhioT1DM.

    Currently, it uses the standard pipeline defined in BaseDataset without
    modifications. If specific cleaning logic is needed in the future,
    implement a DataProcessor and inject it in __init__.
    """
    name = "ohiot1dmMini"

    def __init__(
            self,
            dataset_root: Path,
            config_file: Path,
            global_config_file: Path | None = None,
            patient_metadata_path: Path | None = None,
            logging_dir: Path | None = None
    ):
        super().__init__(
            dataset_root=dataset_root,
            config_file=config_file,
            global_config_file=global_config_file,
            patient_metadata_path=patient_metadata_path,
            logging_dir=logging_dir,
        )

        # No specific processors to inject yet.
        pass