import torch
from .base import Callback


class GenerativeMomentsMetric(Callback):
    """
    Computes statistical moment distances (Mean and Std) between Real and Generated
    distributions over the first N windows of the validation set.
    """

    def __init__(
            self,
            device: torch.device,
            every_n_epochs: int = 5,
            max_batches: int = 20
    ):
        """
        Args:
            device: The device where data should be moved if necessary.
            every_n_epochs: Frequency of calculation.
            max_batches: Number of batches (windows) to analyze.
        """
        self.device = device
        self.every_n_epochs = every_n_epochs
        self.max_batches = max_batches

    def on_epoch_end(self, trainer, model, epoch: int, metrics: dict[str, float]) -> None:
        # 1. Skip if it's not the right epoch
        if epoch % self.every_n_epochs != 0:
            return

        # 2. Safety check: ensure model supports generation and input preparation
        if not hasattr(model, "generate") or not hasattr(model, "prepare_generation_inputs"):
            return

        # 3. Retrieve Validation Loader from Trainer
        val_loader = getattr(trainer, "val_loader", None)
        if val_loader is None:
            print("[Warning] GenerativeMomentsMetric: Val loader not found in Trainer.")
            return

        model.eval()

        # Lists to accumulate all data collected from the N windows
        all_real = []
        all_fake = []

        with torch.no_grad():
            # 4. Iterate over the first N windows
            for i, batch in enumerate(val_loader):
                if i >= self.max_batches:
                    break

                # A. Delegate input preparation to the model
                # The model parses the batch and returns the target (real_X)
                # and a dictionary of arguments compatible with its own generate() method.
                real_X, gen_kwargs = model.prepare_generation_inputs(batch)

                # B. Generate
                # We unpack **gen_kwargs so it matches the specific signature (e.g., c_dyn, c_stat)
                fake_X = model.generate(**gen_kwargs)

                # Handle tuple output (in case generate returns (data, hidden_states))
                if isinstance(fake_X, tuple):
                    fake_X = fake_X[0]

                # C. Move to CPU immediately to avoid OOM and accumulate
                # Assuming shape (Batch, Seq_Len, Features)
                all_real.append(real_X.detach().cpu())
                all_fake.append(fake_X.detach().cpu())

        # 5. Exit if no data was collected
        if not all_real:
            return

        # 6. Concatenation: create two giant tensors with all collected data
        full_real = torch.cat(all_real, dim=0)  # Shape: [Total_Samples, Seq, Feat]
        full_fake = torch.cat(all_fake, dim=0)

        # 7. Compute Moments (Mean and Std)
        # We compute along dim 0 (batch) and 1 (time) to get a metric per Feature.

        # Real Mean & Std
        real_mean = torch.mean(full_real, dim=(0, 1))
        real_std = torch.std(full_real, dim=(0, 1))

        # Generated Mean & Std
        fake_mean = torch.mean(full_fake, dim=(0, 1))
        fake_std = torch.std(full_fake, dim=(0, 1))

        # 8. Compute Distance (L1 Loss between statistics)
        dist_mean = torch.abs(real_mean - fake_mean).mean().item()
        dist_std = torch.abs(real_std - fake_std).mean().item()

        # 9. Logging
        if trainer.logger:
            trainer.logger.log_metrics({
                "moments/dist_mean": dist_mean,
                "moments/dist_std": dist_std
            }, step=epoch, phase="val")

        # Console print for immediate feedback
        print(f"   [Moments Epoch {epoch}] Mean Dist: {dist_mean:.4f} | Std Dist: {dist_std:.4f}")