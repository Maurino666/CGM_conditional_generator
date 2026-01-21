import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """
    Standard Sinusoidal Positional Embedding for the diffusion timestep 't'.
    It allows the network to know "how noisy" the data is currently.
    Matches the implementation used in the original Transformer paper.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (Batch,)
        device = t.device
        half_dim = self.dim // 2

        # Compute the frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Calculate Sin and Cos
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # If dim is odd, pad with a zero (rare edge case, but safe)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))

        return embeddings  # Shape: (Batch, Dim)