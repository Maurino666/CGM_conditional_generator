import torch
import torch.nn as nn
from .interface import BaseDiffusionBackbone
from .layers import TimeEmbedding, ResidualBlock

class DiffWaveBackbone(BaseDiffusionBackbone):
    """
    A 1D ResNet based on DiffWave / WaveNet architecture.
    Optimized for Time Series Generation with limited VRAM.
    """

    def __init__(
            self,
            input_channels: int = 1,
            cond_channels: int = 5,
            residual_channels: int = 64,
            num_layers: int = 12,
            cycle_length: int = 4,
            causal = False,
    ):
        """
        Args:
            input_channels: Dimensions of the target (e.g. 1 for glucose).
            cond_channels: Dimensions of the conditions (insulin + masks + ...).
            residual_channels: Hidden dimension size (64 is good for small GPUs).
            num_layers: Total number of residual blocks.
            cycle_length: How often dilation resets. E.g., if 4: [1, 2, 4, 8, 1, 2, 4, 8...]
            causal: If True, use causal (left-only) padding in dilated convolutions.
        """
        super().__init__()

        self.causal = causal

        # 1. Input Projection (Matches input dim to hidden dim)
        self.input_projection = nn.Conv1d(input_channels, residual_channels, kernel_size=1)

        # 2. Time Embedding Machinery
        self.time_embedding = TimeEmbedding(residual_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(residual_channels, residual_channels),
            nn.SiLU(),  # Swish activation
            nn.Linear(residual_channels, residual_channels)
        )

        # 3. Build Residual Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Dilation grows exponentially: 2^(i % cycle) -> 1, 2, 4, 8...
            dilation = 2 ** (i % cycle_length)

            self.layers.append(
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=dilation,
                    cond_channels=cond_channels,
                    causal=causal,
                )
            )

        # 4. Final Output Block
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, input_channels, kernel_size=1)  # Back to original dim
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting the noise.
        """
        # 1. Embed Time
        t_emb = self.time_embedding(t)  # (B, Res)
        t_emb = self.time_mlp(t_emb)  # (B, Res)

        # 2. Project Input
        x = self.input_projection(x)  # (B, Res, L)

        # 3. Iterate over Residual Layers
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x, t_emb, cond)
            skip_connections.append(skip)

        # 4. Aggregate Skip Connections
        # Summing skips from all layers allows the model to combine short-term (dilation=1)
        # and long-term (dilation=8) features.
        total_skip = torch.sum(torch.stack(skip_connections), dim=0)

        # 5. Final Projection to Output
        output = self.skip_projection(total_skip)
        output = self.output_projection(output)

        return output