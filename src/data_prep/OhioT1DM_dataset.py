from pathlib import Path
from .base_dataset import BaseDataset


class OhioT1DMDataset(BaseDataset):
    """
    Dataset class for OhioT1DM.

    Currently, it uses the standard pipeline defined in BaseDataset without
    modifications. If specific cleaning logic is needed in the future,
    implement a DataProcessor and inject it in __init__.
    """

    def __init__(
            self,
            dataset_root: Path,
            config_file: Path,
            global_config_file: Path | None = None,
            logging_dir: Path | None = None
    ):
        super().__init__(dataset_root, config_file, global_config_file, logging_dir)

        # No specific processors to inject yet.
        pass