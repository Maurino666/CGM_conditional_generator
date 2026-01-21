import sys
import torch
from tqdm.auto import tqdm
from models import BaseTrainableModule
from .loggers import Logger
from .callbacks import Callback


class Trainer:

    callbacks: list[Callback] | None
    max_epochs: int

    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader

    def __init__(
            self,
            device: torch.device,
            logger: Logger | None = None,
            val_check_interval: int = 1,
            log_every_n_steps: int = 50
    ):
        self.device = device
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps

        self.val_check_interval = val_check_interval

    def fit(
            self,
            model: BaseTrainableModule,
            max_epochs: int,
            train_loader,
            val_loader=None,
            callbacks: list[Callback] | None = None,
    ):
        model = model.to(self.device)

        self.max_epochs = max_epochs
        self.callbacks = callbacks if callbacks is not None else []

        self.train_loader = train_loader
        self.val_loader = val_loader

        self._fire_callback("on_train_start", model)

        for epoch in range(1, max_epochs + 1):
            self._fire_callback("on_epoch_start", model, epoch=epoch)

            # --- TRAINING LOOP ---
            train_metrics = self._run_epoch(model, train_loader, phase="train", epoch=epoch)

            # --- VALIDATION LOOP ---
            val_metrics = {}
            # Run the loop if val_loader is present and model supports val_step
            if (
                    val_loader is not None
                    and (epoch % self.val_check_interval == 0)
                    and model.should_validate
            ):
                val_metrics = self._run_epoch(model, val_loader, phase="val", epoch=epoch)

            # --- LOGGING ---
            all_metrics = {**train_metrics, **val_metrics}

            if self.logger:
                self.logger.log_metrics(all_metrics, step=epoch, phase="epoch")

            self._log_to_console(epoch, train_metrics, val_metrics)

            self._fire_callback("on_epoch_end", model, epoch=epoch, metrics=all_metrics)

        self._fire_callback("on_train_end", model)

    def _run_epoch(self, model, loader, phase, epoch):
        is_train = (phase == "train")
        model.train() if is_train else model.eval()

        batch_outputs = []
        desc = f"Epoch {epoch}/{self.max_epochs} [{phase}]"

        pbar = tqdm(
            loader,
            desc=desc,
            leave=True,
            file=sys.stdout,
            mininterval=1.0 # avoiding to update more than 1 time per second
        )
        context = torch.enable_grad() if is_train else torch.no_grad()

        step_count = 0
        with context:
            for i, batch in enumerate(pbar):
                batch = self._move_to_device(batch)

                if is_train:
                    out = model.training_step(batch)
                else:
                    out = model.validation_step(batch)

                # Safety net if out returns None
                if out is None:
                    tqdm.write(f"   [Warning] Batch {i} skipped in epoch {epoch} (returned None).")
                    continue

                # detaching tensor to free memory
                if isinstance(out, dict):
                    safe_out = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
                else:
                    safe_out = out.detach() if isinstance(out, torch.Tensor) else out

                batch_outputs.append(safe_out)

                # logging every n steps
                step_count += 1
                if step_count % self.log_every_n_steps == 0:
                    step_count = 0
                    if isinstance(out, dict):
                        postfix_str = ", ".join(f"{k}={float(v):.4f}" for k, v in out.items())
                        pbar.set_postfix_str(postfix_str)

        return self._aggregate_metrics(batch_outputs, prefix=phase)

    def _aggregate_metrics(self, outputs: list[dict], prefix: str) -> dict:
        """Computes all metrics averages over all batches."""
        if not outputs: return {}

        if isinstance(outputs[0], dict):
            keys = outputs[0].keys()
            avg_metrics = {}

            for k in keys:
                values = []
                for o in outputs:
                    if k in o:
                        val = o[k]
                        if isinstance(val, torch.Tensor):
                            values.append(val.item())
                        else:
                            values.append(val)

                if values:
                    avg_metrics[f"{prefix}/{k}"] = sum(values) / len(values)

            return avg_metrics

        else:
            values = []
            for val in outputs:
                if isinstance(val, torch.Tensor):
                    values.append(val.item())
                else:
                    values.append(val)
                if values:
                    return {f"{prefix}/loss": sum(values) / len(values)}
                #else
                return {}

    def _fire_callback(self, event_name, *args, **kwargs):
        """Fires the event on all subscribed callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, event_name, None)
            if method:
                method(self, *args, **kwargs)

    def _log_to_console(self, epoch, train_metrics, val_metrics):
        """
        Prints summary at the end of each epoch.
        """
        parts = []

        if train_metrics:
            t_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            parts.append(f"Train[{t_str}]")

        if val_metrics:
            v_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            parts.append(f"Val[{v_str}]")

        print(f"Epoch {epoch:03d}: " + " | ".join(parts))

    def _move_to_device(self, batch):
        """Recursively move a batch to the given device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v) for v in batch]
        return batch