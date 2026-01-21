import sys
from pathlib import Path  ### NEW: Importiamo Path
from typing import Any
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.models import BaseTrainableModule

from collections.abc import Mapping, Sequence
from torch import Tensor


def _move_to_device(batch, device):
    """Recursively move a batch to the given device."""
    if isinstance(batch, Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return type(batch)(_move_to_device(x, device) for x in batch)
    return batch


def _aggregate_metrics(batch_outputs: list[Any]) -> dict[str, float]:
    """Aggregate per-batch outputs into per-epoch metrics."""
    if not batch_outputs:
        return {}
    first = batch_outputs[0]

    # Caso scalare (es. loss singola)
    if not isinstance(first, dict):
        values = [float(v) for v in batch_outputs]
        avg = sum(values) / len(values)
        return {"loss": avg}

    # Caso dizionario (es. g_loss, d_loss)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for out in batch_outputs:
        if not isinstance(out, dict):
            out = {"loss": float(out)}
        for k, v in out.items():
            fv = float(v)
            sums[k] = sums.get(k, 0.0) + fv
            counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums.keys()}


def _log_visualizations(model, fixed_batch, device, writer, epoch):
    """
    Genera un confronto Real vs Fake su TensorBoard.
    """
    if not hasattr(model, "generate"):
        return

    model.eval()
    with torch.no_grad():
        # Gestione input: supporta tuple [target, condition] o tensori singoli
        if isinstance(fixed_batch, (list, tuple)) and len(fixed_batch) >= 2:
            c_fixed = fixed_batch[1].to(device)
            y_real = fixed_batch[0].cpu().numpy()

            generated = model.generate(c_fixed)
            if isinstance(generated, tuple):
                y_fake = generated[0].cpu().numpy()
            else:
                y_fake = generated.cpu().numpy()
        else:
            return

            # Plot del primo soggetto del batch
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_real[0], label='Real', color='black', alpha=0.7, linewidth=2)
        ax.plot(y_fake[0], label='Generated', color='red', linestyle='--', linewidth=2)

        ax.set_title(f"Epoch {epoch} - Visual Check")
        ax.legend()
        ax.grid(True, alpha=0.3)

        writer.add_figure("Visual/Real_vs_Synth", fig, epoch)
        plt.close(fig)


def train_module(
        model: BaseTrainableModule,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int,
        device: torch.device,
        tensorboard_dir: Path | None = None  ### NEW: Type hint aggiornato a Path
) -> dict[str, Any]:
    # Move model to device
    model = model.to(device)

    # --- Setup TensorBoard ---
    writer = None
    fixed_vis_batch = None

    if tensorboard_dir:
        # Assicuriamoci che sia un oggetto Path
        if isinstance(tensorboard_dir, str):
            tensorboard_dir = Path(tensorboard_dir)

        # Usiamo l'operatore / di pathlib invece di os.path.join
        phase_suffix = getattr(model, "phase", "")
        log_dir = tensorboard_dir / phase_suffix

        # SummaryWriter vuole una stringa
        writer = SummaryWriter(log_dir=str(log_dir))

        tqdm.write(f"   [Logging] TensorBoard attivo in: {log_dir}", file=sys.stdout)

        # Recupero batch fisso per visualizzazione
        loader_for_vis = val_loader if val_loader is not None else train_loader
        try:
            fixed_vis_batch = next(iter(loader_for_vis))
        except StopIteration:
            pass
    # -------------------------

    history: dict[str, list[dict[str, float]]] = {"train": []}
    do_validation = val_loader is not None and hasattr(model, "validation_step")
    if do_validation:
        history["val"] = []

    tqdm.write(f"Training is starting on device = {device}", file=sys.stdout)

    for epoch in range(1, num_epochs + 1):
        # ----------------------------
        # TRAINING
        # ----------------------------
        model.train()
        train_batch_outputs: list[Any] = []

        with tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{num_epochs} [train]",
                unit="batch",
                leave=True,
                file=sys.stdout,
        ) as progress_bar:
            for batch in progress_bar:
                batch_on_device = _move_to_device(batch, device)
                batch_out = model.training_step(batch_on_device)
                train_batch_outputs.append(batch_out)

                if isinstance(batch_out, dict):
                    postfix_str = ", ".join(f"{k}={float(v):.4f}" for k, v in batch_out.items())
                    progress_bar.set_postfix_str(postfix_str)
                else:
                    loss_value = float(batch_out)
                    progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        epoch_train_metrics = _aggregate_metrics(train_batch_outputs)
        history["train"].append(epoch_train_metrics)

        if writer:
            for k, v in epoch_train_metrics.items():
                writer.add_scalar(f"Train/{k}", v, epoch)

        # ----------------------------
        # VALIDATION
        # ----------------------------
        epoch_val_metrics: dict[str, float] | None = None

        if do_validation:
            model.eval()
            val_batch_outputs: list[Any] = []

            with torch.no_grad():
                for batch in val_loader:
                    batch_on_device = _move_to_device(batch, device)
                    batch_out = model.validation_step(batch_on_device)
                    val_batch_outputs.append(batch_out)

            epoch_val_metrics = _aggregate_metrics(val_batch_outputs)
            history["val"].append(epoch_val_metrics)

            if writer:
                for k, v in epoch_val_metrics.items():
                    writer.add_scalar(f"Val/{k}", v, epoch)

        # ----------------------------
        # VISUAL CHECK (Log Image)
        # ----------------------------
        if writer and fixed_vis_batch is not None:
            # Salva l'immagine ogni 5 epoche o all'ultima
            _log_visualizations(model, fixed_vis_batch, device, writer, epoch)

        # ----------------------------
        # SUMMARY PRINT
        # ----------------------------
        def _metrics_to_str(metrics: dict[str, float]) -> str:
            return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else "-"

        tqdm.write("", file=sys.stdout)
        train_str = _metrics_to_str(epoch_train_metrics)

        if epoch_val_metrics is not None:
            val_str = _metrics_to_str(epoch_val_metrics)
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} - train: {train_str}  |  val: {val_str}",
                file=sys.stdout,
            )
        else:
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} - train: {train_str}",
                file=sys.stdout,
            )

    if writer:
        writer.close()

    return history