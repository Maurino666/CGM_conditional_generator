import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for fixed-length time series windows.

    Each item is a single window (sequence) of shape (seq_len, num_features).
    """

    def __init__(self, sequences: np.ndarray) -> None:
        """
        :param sequences: Numpy array of shape
                          (num_windows, seq_len, num_features).
        """
        if sequences.ndim != 3:
            raise ValueError(
                f"`sequences` must be a 3D array (num_windows, seq_len, num_features), "
                f"got shape {sequences.shape}"
            )

        # Store as float32 for PyTorch
        self.sequences = sequences.astype(np.float32)

    def __len__(self) -> int:
        """Return the total number of windows."""
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """
        Return a single window as a torch.Tensor.

        Output shape: (seq_len, num_features)
        """
        x = self.sequences[idx]  # (seq_len, num_features)
        return torch.from_numpy(x)
