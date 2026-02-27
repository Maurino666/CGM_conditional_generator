from typing import Any
import torch.nn as nn

from .base_diffusion_module import BaseDiffusionModule
from .backbone import DiffWaveBackbone


class DiffWaveDiffusionModule(BaseDiffusionModule):
    """
    Concrete implementation of Diffusion using the DiffWave (CNN) backbone.
    """

    def __init__(
            self,
            # Common params
            input_dim: int,
            cond_dim: int,
            # DiffWave specific params
            residual_channels: int = 64,
            num_layers: int = 12,
            cycle_length: int = 4,
            causal: bool = False,
            # Base params (with defaults)
            **kwargs
    ):
        # Save specific config first
        self.residual_channels = residual_channels
        self.num_layers = num_layers
        self.cycle_length = cycle_length
        self.causal = causal

        # Init base class
        super().__init__(input_dim=input_dim, cond_dim=cond_dim, **kwargs)

    def _build_backbone(self) -> nn.Module:
        """
        Implementation of the factory method.
        Returns the DiffWave CNN.
        """
        return DiffWaveBackbone(
            input_channels=self.input_dim,
            cond_channels=self.cond_dim,
            residual_channels=self.residual_channels,
            num_layers=self.num_layers,
            cycle_length=self.cycle_length,
            causal=self.causal,
        )

    def get_config(self) -> dict[str, Any]:
        # Extend base config with specific params
        cfg = super().get_config()
        cfg.update({
            "residual_channels": self.residual_channels,
            "num_layers": self.num_layers,
            "cycle_length": self.cycle_length,
            # Used for reconstruction from checkpoint
            "architecture_type": "diffwave"
        })
        return cfg