import torch
import matplotlib.pyplot as plt
from .base import Callback


class GenerativeVisualizer(Callback):
    """
    Callback for visualizing generative model outputs during training.

    It performs inference on a fixed validation batch and logs a comparison
    plot (Real vs Generated) to the Trainer's logger.
    """

    def __init__(self, fixed_batch, device: torch.device, every_n_epochs: int = 5):
        """
        Args:
            fixed_batch: A tuple or tensor structure representing a fixed reference batch.
            device: The device where the model is running.
            every_n_epochs: Frequency of logging.
        """
        self.fixed_batch = fixed_batch
        self.device = device
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, model, epoch: int, metrics: dict[str, float]) -> None:
        # Skip if it's not the right epoch
        if epoch % self.every_n_epochs != 0:
            return

        # Safety check: ensure model supports generation and input preparation
        if not hasattr(model, "generate") or not hasattr(model, "prepare_generation_inputs"):
            return

        model.eval()
        with torch.no_grad():

            # 1. Prepare Inputs using the Model's Adapter
            # The model knows how to unpack fixed_batch and move necessary
            # conditions to the correct device.
            real_X, gen_kwargs = model.prepare_generation_inputs(self.fixed_batch)

            # 2. Generate
            # Pass the arguments exactly as the model expects them
            out = model.generate(**gen_kwargs)

            # Handle tuple output
            if isinstance(out, tuple):
                fake_X = out[0]
            else:
                fake_X = out

            # Move results to CPU and convert to Numpy for plotting
            # real_X might still be on GPU depending on prepare_generation_inputs implementation
            if isinstance(real_X, torch.Tensor):
                real_X = real_X.detach().cpu().numpy()

            fake_X = fake_X.detach().cpu().numpy()

        # 3. Create Plot (Comparison of the first subject in the batch)
        # We assume the shape is (Batch, Seq_Len, Features)
        # We take the first feature (usually Glucose) of the first sample.
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot Real Data
        ax.plot(real_X[0, :, 0], label='Real Glucose', color='black', alpha=0.7, linewidth=1.5)

        # Plot Generated Data
        ax.plot(fake_X[0, :, 0], label='Generated Glucose', color='orange', linestyle='--', linewidth=1.5)

        ax.set_title(f"Generative Check - Epoch {epoch}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Log to TensorBoard (via Trainer's Logger)
        if trainer.logger:
            trainer.logger.log_figure("Visual/Real_vs_Fake", fig, step=epoch)
            # Optional: force flush to disk to avoid missing plots in UI
            if hasattr(trainer.logger, 'writer'):
                trainer.logger.writer.flush()

        plt.close(fig)  # Close figure to free memory