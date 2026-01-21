import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict

from .base import Logger


class WandBLogger(Logger):
    """
    Concrete implementation of Logger using Weights & Biases (WandB).

    WandB offers superior visualization consistency compared to TensorBoard,
    preserving image history and allowing interactive data exploration.
    """

    def __init__(
            self,
            project_name: str,
            run_name: str | None = None,
            config: Dict[str, Any] | None = None,
            log_dir: Path | None = None
    ):
        """
        Initializes the WandB run.

        Args:
            project_name: The name of the project in the WandB dashboard.
            run_name: A specific name for this training run (optional).
            config: A dictionary of hyperparameters to save for comparison.
            log_dir: Directory where local metadata is stored (optional).
        """
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            dir=str(log_dir) if log_dir else None,
            reinit="finish_previous"
        )
        print(f"   [Logger] WandB initialized: {self.run.name} (Project: {project_name})")

    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = "train") -> None:
        """
        Logs scalar metrics to WandB.

        WandB uses a flat dictionary structure with '/' separators for grouping,
        similar to TensorBoard but more flexible in the UI.
        """
        # Prepare dictionary with phase prefix (e.g., "train/loss", "val/mse")
        log_data = {f"{phase}/{k}": v for k, v in metrics.items()}

        # Explicitly setting the step ensures alignment across different metrics
        self.run.log(log_data)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """
        Logs a matplotlib figure as an image.

        WandB automatically converts matplotlib figures to interactive images
        or static PNGs depending on complexity. Unlike TensorBoard, it preserves history.
        """
        # Create a WandB Image object from the matplotlib figure
        image = wandb.Image(figure, caption=f"{tag} at step {step}")

        # Log it. The tag serves as the panel name in the dashboard.
        self.run.log({tag: image})

    def close(self) -> None:
        """
        Finishes the run and uploads final artifacts.
        """
        print("   [Logger] Finishing WandB run...")
        self.run.finish()