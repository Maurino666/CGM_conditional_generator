import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("This callback requires scikit-learn. Install it via: pip install scikit-learn")

from .base import Callback


class GenerativePCAVisualizer(Callback):
    """
    Callback to visualize the distribution of Real vs Generated data using PCA.

    It collects a subset of validation data and generated samples, flattens the
    time-series sequences, and projects them onto a 2D plane.

    Interpretation:
    - Good Training: The 'Real' and 'Generated' point clouds overlap significantly.
    - Mode Collapse: The 'Generated' points are clustered in a single small area.
    - Bad Training: The two clouds are completely separated.
    """

    def __init__(
            self,
            device: torch.device,
            every_n_epochs: int = 5,
            max_batches: int = 20,
            n_components: int = 2
    ):
        """
        Args:
            device: Device for computation.
            every_n_epochs: Frequency of plotting.
            max_batches: Number of batches to collect for the PCA fit (higher = better approximation).
            n_components: Number of PCA components (usually 2 for visualization).
        """
        self.device = device
        self.every_n_epochs = every_n_epochs
        self.max_batches = max_batches
        self.n_components = n_components

    def on_epoch_end(self, trainer, model, epoch: int, metrics: dict[str, float]) -> None:
        # 1. Skip checks
        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(model, "generate") or not hasattr(model, "prepare_generation_inputs"):
            return

        val_loader = getattr(trainer, "val_loader", None)
        if val_loader is None:
            return

        model.eval()

        # Lists to accumulate data
        all_real = []
        all_fake = []

        # 2. Data Collection Loop
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= self.max_batches:
                    break

                # Use the standard adapter
                real_X, gen_kwargs = model.prepare_generation_inputs(batch)
                fake_X = model.generate(**gen_kwargs)

                if isinstance(fake_X, tuple):
                    fake_X = fake_X[0]

                # Move to CPU immediately
                all_real.append(real_X.detach().cpu())
                all_fake.append(fake_X.detach().cpu())

        if not all_real:
            return

        # 3. Concatenate and Flatten
        # Original Shape: (Total_Samples, Seq_Len, Features)
        full_real = torch.cat(all_real, dim=0)
        full_fake = torch.cat(all_fake, dim=0)

        N, Seq, Feat = full_real.shape

        # Flattening: (N, Seq * Feat)
        # We treat the entire time series as a single high-dimensional point vector.
        real_flat = full_real.reshape(N, -1).numpy()
        fake_flat = full_fake.reshape(N, -1).numpy()

        # 4. Perform PCA
        # Combine data to fit PCA on the joint distribution
        data_combined = np.concatenate([real_flat, fake_flat], axis=0)

        # Fit PCA
        pca = PCA(n_components=self.n_components)
        pca_results = pca.fit_transform(data_combined)

        # Split back into Real and Fake
        real_pca = pca_results[:N]
        fake_pca = pca_results[N:]

        # 5. Plotting
        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter Plot
        # Alpha is important to see density
        ax.scatter(real_pca[:, 0], real_pca[:, 1], c='black', alpha=0.2, label='Real', s=10)
        ax.scatter(fake_pca[:, 0], fake_pca[:, 1], c='orange', alpha=0.3, label='Generated', s=10)

        ax.set_title(f"PCA Analysis - Epoch {epoch}")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Log to TensorBoard
        if trainer.logger:
            trainer.logger.log_figure("Visual/PCA_Distribution", fig, step=epoch)
            if hasattr(trainer.logger, 'writer'):
                trainer.logger.writer.flush()

        plt.close(fig)