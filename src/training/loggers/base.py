from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Logger(ABC):
    """
    Abstract base class for experiment loggers.

    This interface defines the standard methods that any logger (TensorBoard,
    WandB, CSV, etc.) must implement to be compatible with the Trainer.
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int, phase: str = "train") -> None:
        """
        Logs a dictionary of scalar metrics.

        Args:
            metrics (dict[str, float]): Dictionary of metric names and values
                                        (e.g., {'loss': 0.5, 'accuracy': 0.9}).
            step (int): The current global step or epoch.
            phase (str): The training phase ('train', 'val', 'test'), used as a prefix.
        """
        pass

    @abstractmethod
    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """
        Logs a matplotlib figure (image).

        Args:
            tag (str): The identifier for the figure (e.g., 'Visual/Real_vs_Fake').
            figure (plt.Figure): The matplotlib figure object to log.
            step (int): The current global step or epoch.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Finalizes the logger and flushes any pending data to disk.
        """
        pass