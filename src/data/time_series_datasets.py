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



class ConditionalTimeSeriesDataset(Dataset):
    """Dataset for conditional time series (target + conditional features)."""

    def __init__(self, targets: np.ndarray, conditionals: np.ndarray) -> None:
        """
        Parameters
        ----------
        targets : np.ndarray
            Target sequences, shape (num_windows, seq_len, target_dim).
        conditionals : np.ndarray
            Conditional sequences, shape (num_windows, seq_len, cond_dim).
        """
        if targets.ndim != 3 or conditionals.ndim != 3:
            raise ValueError("targets and conditionals must be 3D arrays.")
        if targets.shape[0] != conditionals.shape[0]:
            raise ValueError("targets and conditionals must have same num_windows.")
        if targets.shape[1] != conditionals.shape[1]:
            raise ValueError("targets and conditionals must have same seq_len.")

        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.conditionals = torch.as_tensor(conditionals, dtype=torch.float32)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        Tuple[Tensor, Tensor]
            (target_seq, conditional_seq), each with shape (seq_len, dim).
        """
        return self.targets[idx], self.conditionals[idx]
