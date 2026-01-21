from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data_prep import AZT1D2025Dataset, HUPA_UCMDataset, OhioT1DMDataset
from src.windowing.builder import ConditionalWindowBuilder, ConditionalWindowingConfig
from src.reconstruction.reconstructor import WindowReconstructor, ReconstructionConfig

# Your training utilities / models
from models import ConditionalTimeGanModule  # example
from models import train_module               # example


def generate_val_target_windows(
    model: torch.nn.Module,
    *,
    c_val: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Model-agnostic placeholder:
    returns y_hat windows aligned with c_val.

    IMPORTANT:
      - this function must not use DataLoader order (shuffle).
      - it must follow the deterministic ordering in pack.c_val.

    Adapt this to your model API:
      - some models take (noise, cond)
      - some models take (cond) and sample noise internally
      - some return full (y, cond) etc.
    """
    model.eval()

    c = torch.as_tensor(c_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Example: assume model has a method generate_from_conditionals(c) -> (N, seq_len, 1)
        y_hat = model.generate(c)  # type: ignore[attr-defined]

    return y_hat.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    target_col = "glucose"
    cond_cols = ["basal_rate"]

    # 1) Instantiate and clean BaseDatasets (list[df] world)
    ds1 = AZT1D2025Dataset(
        Path("../datasets/AZT1D2025/CGM Records"),
        Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    ds2 = HUPA_UCMDataset(
        Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )
    ds3 = OhioT1DMDataset(
        Path("../datasets/OhioT1DMmini"),
        Path("../datasets/OhioT1DMmini/ohiot1dmmini.yaml"),
        logging_dir=Path("../datasets/OhioT1DMmini/prep_logs"),
    )

    for ds in (ds1, ds2, ds3):
        ds.clean_data()

    # 2) Build windows + loaders (windowing world)
    cfg = ConditionalWindowingConfig(
        train_seq_len=228,
        train_step=12,
        val_seq_len=228,
        val_step=228,
        val_ratio=0.2,
        split_by="subject",
        random_state=1,
        batch_size=64,
        num_workers=4,
        normalize=[target_col] + cond_cols,
        freq_minutes=5,
    )

    builder = ConditionalWindowBuilder(cfg)
    pack, train_loader, val_loader = builder.build_from_datasets(
        [ds1, ds2, ds3],
        cond_cols=cond_cols,
        target_col=target_col,
    )

    # 3) Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalTimeGanModule(
        cond_dim=len(cond_cols),
        hidden_dim=64,
        num_layers=1,
        g_steps_per_iter=1,
    ).to(device)

    # Example training phases (adapt to your module API)
    model.set_phase("ae")
    train_module(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=3, device=device)

    model.set_phase("sup")
    train_module(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=3, device=device)

    model.set_phase("adv")
    train_module(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=3, device=device)

    # 4) Generate synthetic windows for VAL (default evaluation split)
    y_hat_val = generate_val_target_windows(model, c_val=pack.c_val, device=device)

    # 5) Reconstruct list[df] for VAL
    recon = WindowReconstructor(
        ReconstructionConfig(
            target_col=pack.target_col,
            cond_cols=pack.cond_cols,
            synth_col="glucose_synth",
            include_true_target=True,
        )
    )
    synthetic_val_dfs = recon.reconstruct_subject_dfs(
        templates=pack.val_templates,
        meta=pack.meta_val,
        c_windows=pack.c_val,
        y_hat_windows=y_hat_val,
    )

    # 6) Evaluate (your existing metrics expect list[pd.DataFrame])
    # from src.metrics_pipeline.evaluate import evaluate_dataset
    # results = evaluate_dataset(synthetic_val_dfs, target_col=target_col, ...)
    # print(results)

    print(f"Reconstructed {len(synthetic_val_dfs)} validation subject dfs.")


if __name__ == "__main__":
    main()
