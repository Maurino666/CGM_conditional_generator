from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .base import Logger


class TensorBoardLogger(Logger):
    """
    Concrete implementation of BaseLogger using PyTorch's TensorBoard SummaryWriter.
    """

    def __init__(self, log_dir: Path):
        """
        Initializes the TensorBoard writer.

        Args:
            log_dir (Path): The directory where event files will be saved.
        """
        # Convert Path to string because SummaryWriter expects a string path
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"   [Logger] TensorBoard initialized at: {self.log_dir}")

    def log_metrics(self, metrics: dict[str, float], step: int, phase: str = "train") -> None:
        """
        Logs scalar metrics to TensorBoard grouping them by phase.
        Example tag produced: 'train/loss' or 'val/mse'.
        """
        for key, value in metrics.items():
            # Combine phase and key to create a structured tag (e.g., "train/g_loss")
            tag = f"{phase}/{key}"
            self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """
        Logs a matplotlib figure to TensorBoard.
        """
        self.writer.add_figure(tag, figure, global_step=step)

    def close(self) -> None:
        """
        Closes the SummaryWriter.
        """
        self.writer.close()