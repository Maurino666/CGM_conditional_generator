import sys
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models import BaseTrainableModule


def _aggregate_metrics(batch_outputs: list[Any]) -> dict[str, float]:
    """
    Aggregate per-batch outputs (float or dict[str, float]) into
    per-epoch metrics.

    - If elements are floats / scalari: returns {"loss": avg_loss}.
    - If elements are dicts: media per chiave su tutti i batch.
    """
    if not batch_outputs:
        return {}

    first = batch_outputs[0]

    # Caso 1: il modello restituisce un float / scalar (es. VAE)
    if not isinstance(first, dict):
        values = [float(v) for v in batch_outputs]
        avg = sum(values) / len(values)
        return {"loss": avg}

    # Caso 2: il modello restituisce dict[str, float] (es. TimeGAN)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for out in batch_outputs:
        if not isinstance(out, dict):
            # per sicurezza, converto eventuali float "nudi" sotto chiave "loss"
            out = {"loss": float(out)}  # type: ignore[assignment]
        for k, v in out.items():
            fv = float(v)
            sums[k] = sums.get(k, 0.0) + fv
            counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums.keys()}


def train_module(
    model: BaseTrainableModule,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    num_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Generic training loop for module-like models.

    The model is expected to implement:
      - training_step(batch: Tensor) -> float | dict[str, float]
          Performs a full optimization step (forward + loss + backward + optimizer.step)
          and returns either:
            * a scalar loss value (float), or
            * a dict of named losses/metrics.
      - validation_step(batch: Tensor) -> float | dict[str, float] (OPTIONAL)
          Computes validation metrics on the batch (no backward, no optimizer.step).

    Parameters
    ----------
    model : BaseTrainableModule
        Model object implementing the interface above
        (e.g. CgmVaeModule, TimeGanModule).
    train_loader : DataLoader
        DataLoader yielding training batches (tensors of sequences).
    val_loader : DataLoader | None
        DataLoader yielding validation batches, or None to skip validation.
    num_epochs : int
        Number of training epochs.
    device : torch.device
        Device to use ("cpu" or "cuda").

    Returns
    -------
    history : dict[str, Any]
        Dictionary containing per-epoch metrics:
          - history["train"]: list[dict[str, float]]  (per-epoch training metrics)
          - history["val"]:   list[dict[str, float]]  (per-epoch validation metrics, if available)

        For single-loss models, each dict has only {"loss": value}.
        For multi-loss models (e.g. TimeGAN), the dict may contain
        keys like "g_loss", "d_loss", "er_loss", etc.
    """
    # Move model to device
    model = model.to(device)

    history: dict[str, list[dict[str, float]]] = {"train": []}
    do_validation = val_loader is not None and hasattr(model, "validation_step")
    if do_validation:
        history["val"] = []

    tqdm.write(f"Training is starting on device = {device}", file=sys.stdout)

    for epoch in range(1, num_epochs + 1):
        # ----------------------------
        # TRAINING PHASE
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
                # Here we assume that each batch is directly a Tensor of inputs.
                # If later you have (inputs, labels), you can unpack here.
                x: Tensor = batch.to(device, non_blocking=True)

                # The model performs the optimization step internally.
                batch_out = model.training_step(x)
                train_batch_outputs.append(batch_out)

                # Update progress bar:
                # - if float: show "loss=..."
                # - if dict: show "k1=v1, k2=v2, ..."
                if isinstance(batch_out, dict):
                    postfix_str = ", ".join(
                        f"{k}={float(v):.4f}" for k, v in batch_out.items()
                    )
                    progress_bar.set_postfix_str(postfix_str)
                else:
                    loss_value = float(batch_out)
                    progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        epoch_train_metrics = _aggregate_metrics(train_batch_outputs)
        history["train"].append(epoch_train_metrics)

        # ----------------------------
        # VALIDATION PHASE (OPTIONAL)
        # ----------------------------
        epoch_val_metrics: dict[str, float] | None = None

        if do_validation:
            model.eval()
            val_batch_outputs: list[Any] = []

            with torch.no_grad():
                for batch in val_loader:  # type: ignore[arg-type]
                    x = batch.to(device, non_blocking=True)
                    batch_out = model.validation_step(x)  # type: ignore[attr-defined]
                    val_batch_outputs.append(batch_out)

            epoch_val_metrics = _aggregate_metrics(val_batch_outputs)
            history["val"].append(epoch_val_metrics)

        # ----------------------------
        # EPOCH SUMMARY
        # ----------------------------
        def _metrics_to_str(metrics: dict[str, float]) -> str:
            return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else "-"

        tqdm.write("", file=sys.stdout)  # blank line for readability
        train_str = _metrics_to_str(epoch_train_metrics)

        if epoch_val_metrics is not None:
            val_str = _metrics_to_str(epoch_val_metrics)
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} - "
                f"train: {train_str}  |  val: {val_str}",
                file=sys.stdout,
            )
        else:
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} - "
                f"train: {train_str}",
                file=sys.stdout,
            )

    return history
