from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data import create_conditional_dataloaders, minmax_scale_conditional
from .packs import ConditionalWindowPack
from .utils import WindowMetadata, build_sliding_windows_conditional


@dataclass(frozen=True)
class ConditionalWindowingConfig:
    """
    Configuration for building conditional windows and loaders.
    """
    train_seq_len: int
    train_step: int
    val_seq_len: int | None = None
    val_step: int | None = None

    val_ratio: float = 0.2
    split_by: str = "subject"         # "subject" | "time"
    random_state: int | None = None
    max_missing_ratio: float = 0.0

    # Reconstruction assumptions
    freq_minutes: int = 5

    # Normalization (feature names)
    # Example: normalize = ["glucose", "basal_rate", ...]
    normalize: list[str] | None = None

    # Dataloader
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True


class ConditionalWindowBuilder:
    """
    Build conditional windows for one or more BaseDataset-like objects.

    Design goals:
      - BaseDataset stays in "list[pd.DataFrame]" world.
      - This builder converts list[df] -> windows + metadata + loaders.
      - Produces a ConditionalWindowPack (deterministic, stable interface).
    """

    def __init__(self, config: ConditionalWindowingConfig) -> None:
        self.cfg = config

    def build_from_datasets(
        self,
        datasets: list[object],
        *,
        cond_cols: list[str],
        target_col: str,
    ) -> tuple[ConditionalWindowPack, DataLoader, DataLoader]:
        """
        Build train/val windows from multiple datasets, assigning global subject ids.

        Parameters
        ----------
        datasets:
            Objects exposing:
              - clean_data()
              - split(val_ratio, split_by, random_state) -> (train_dfs, val_dfs, train_ids, val_ids)
        cond_cols:
            Conditional columns to extract.
        target_col:
            Target column name (must be consistent across datasets).

        Returns
        -------
        pack, train_loader, val_loader
        """
        cfg = self.cfg

        val_seq_len = cfg.val_seq_len if cfg.val_seq_len is not None else cfg.train_seq_len
        val_step = cfg.val_step if cfg.val_step is not None else cfg.train_step

        all_y_train: list[np.ndarray] = []
        all_c_train: list[np.ndarray] = []
        all_y_val: list[np.ndarray] = []
        all_c_val: list[np.ndarray] = []

        all_meta_train: list[WindowMetadata] = []
        all_meta_val: list[WindowMetadata] = []

        train_templates: dict[int, pd.DataFrame] = {}
        val_templates: dict[int, pd.DataFrame] = {}

        # Global id assignment across datasets
        global_offset = 0

        for ds in datasets:
            # 1) Split in list[df] space (BaseDataset layer)
            train_dfs, val_dfs, train_ids_local, val_ids_local = ds.split(
                val_ratio=cfg.val_ratio,
                split_by=cfg.split_by,
                random_state=cfg.random_state,
            )

            # 2) Create local->global id mapping
            # train_ids_local/val_ids_local are "global within dataset"; we shift them by an offset.
            # Important: the ids we store in metadata must match templates dict keys.
            train_ids_global = [int(i) + global_offset for i in train_ids_local]
            val_ids_global = [int(i) + global_offset for i in val_ids_local]

            # 3) Store templates (per-subject df) for reconstruction
            # We keep the full df segments corresponding to each split.
            # Keys are global subject ids.
            for df, gid in zip(train_dfs, train_ids_global, strict=False):
                train_templates[gid] = df
            for df, gid in zip(val_dfs, val_ids_global, strict=False):
                val_templates[gid] = df

            # 4) Build conditional windows + metadata for train split
            y_tr, c_tr, meta_tr = build_sliding_windows_conditional(
                all_data=train_dfs,
                seq_len=cfg.train_seq_len,
                step=cfg.train_step,
                target_col=target_col,
                cond_cols=cond_cols,
                ids=train_ids_global,  # ensures metadata uses *global* ids
                max_missing_ratio=cfg.max_missing_ratio,
            )

            # 5) Build conditional windows + metadata for val split
            y_va, c_va, meta_va = build_sliding_windows_conditional(
                all_data=val_dfs,
                seq_len=val_seq_len,
                step=val_step,
                target_col=target_col,
                cond_cols=cond_cols,
                ids=val_ids_global,
                max_missing_ratio=cfg.max_missing_ratio,
            )

            all_y_train.append(y_tr)
            all_c_train.append(c_tr)
            all_meta_train.extend(meta_tr)

            all_y_val.append(y_va)
            all_c_val.append(c_va)
            all_meta_val.extend(meta_va)

            # 6) Update offset: we need the number of subjects in the dataset, not only train/val counts.
            # We assume datasets expose ds.all_data (list of subject dfs). If not, provide a ds.num_subjects property.
            num_subjects_in_ds = len(ds.all_data)  # type: ignore[attr-defined]
            global_offset += int(num_subjects_in_ds)

        # 7) Concatenate windows
        y_train = np.concatenate(all_y_train, axis=0) if all_y_train else np.empty((0, cfg.train_seq_len, 1), dtype=np.float32)
        c_train = np.concatenate(all_c_train, axis=0) if all_c_train else np.empty((0, cfg.train_seq_len, len(cond_cols)), dtype=np.float32)
        y_val = np.concatenate(all_y_val, axis=0) if all_y_val else np.empty((0, val_seq_len, 1), dtype=np.float32)
        c_val = np.concatenate(all_c_val, axis=0) if all_c_val else np.empty((0, val_seq_len, len(cond_cols)), dtype=np.float32)

        # 8) Optional normalization (fit on train, apply to train+val)
        scaling_params = None
        if cfg.normalize:
            y_train, c_train, y_val, c_val, scaling_params = minmax_scale_conditional(
                y_train=y_train,
                c_train=c_train,
                y_val=y_val,
                c_val=c_val,
                target_feature=target_col,
                cond_features=cond_cols,
                normalize=cfg.normalize,
            )

        # 9) Build loaders (these may shuffle, pack remains deterministic)
        train_loader, val_loader = create_conditional_dataloaders(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            batch_size=cfg.batch_size,
            shuffle_train=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        )

        pack = ConditionalWindowPack(
            y_train=y_train,
            c_train=c_train,
            y_val=y_val,
            c_val=c_val,
            meta_train=all_meta_train,
            meta_val=all_meta_val,
            train_templates=train_templates,
            val_templates=val_templates,
            target_col=target_col,
            cond_cols=cond_cols,
            freq_minutes=cfg.freq_minutes,
            split_by=cfg.split_by,
            extra={
                "train_seq_len": cfg.train_seq_len,
                "train_step": cfg.train_step,
                "val_seq_len": val_seq_len,
                "val_step": val_step,
                "scaling_params": scaling_params,
            },
        )

        return pack, train_loader, val_loader
