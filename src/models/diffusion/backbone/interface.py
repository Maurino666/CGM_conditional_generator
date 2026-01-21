import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseDiffusionBackbone(nn.Module, ABC):
    """
    Abstract Interface for Diffusion Backbones.
    Any network used in the DiffusionModule must implement this forward signature.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, In_Channels, Seq_Len) - The noisy input.
            t: (Batch,) - The timestep integers.
            cond: (Batch, Cond_Channels, Seq_Len) - The conditioning features.

        Returns:
            torch.Tensor: (Batch, In_Channels, Seq_Len) - The predicted noise.
        """
        pass