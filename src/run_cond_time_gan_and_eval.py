from __future__ import annotations


from pathlib import Path

import numpy as np
import torch
import yaml
import pandas as pd

# Import Datasets
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset#, OhioT1DMDataset
# Import Windowing
from windowing import ConditionalWindowBuilder, ConditionalWindowingConfig
# Import Reconstruction
from reconstruction import WindowReconstructor, ReconstructionConfig
# Import Model and Trainer
from models import ConditionalTimeGanModule, train_module


# Optional: Import Evaluation Pipeline if available
# from metrics_pipeline.evaluate import evaluate_reconstructed_data

def save_results_as_csv_folder(
        dfs: list[pd.DataFrame],
        output_dir: Path,
        prefix: str = "synthetic_subject"
) -> None:
    """
    Saves a list of DataFrames as individual CSV files in a specific folder.

    Args:
        dfs: List of reconstructed DataFrames.
        output_dir: Root directory where the 'csv_data' folder will be created.
        prefix: Filename prefix (e.g., 'synthetic_subject').
    """
    # Create a dedicated sub-folder for CSVs
    csv_dir = output_dir / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Saving {len(dfs)} CSV files to: {csv_dir}")

    for i, df in enumerate(dfs):
        # We assume the index (timestamp) is important, so index=True
        filename = f"{prefix}_{i:03d}.csv"  # e.g., synthetic_subject_000.csv
        save_path = csv_dir / filename

        df.to_csv(save_path, index=True)

    print("   CSV saving complete.")

def generate_val_target_windows(
        model: torch.nn.Module,
        *,
        c_val: np.ndarray,
        device: torch.device,
) -> np.ndarray:
    """
    Generates synthetic target windows (y_hat) based on validation conditions (c_val).

    Args:
        model: Trained generative model.
        c_val: Numpy array of validation condition windows.
        device: Torch device.

    Returns:
        np.ndarray: Synthetic target windows.
    """
    model.eval()
    # Convert numpy to tensor
    c = torch.as_tensor(c_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Handle models returning tuples (e.g., (output, hidden_state))
        out = model.generate(c)

        if isinstance(out, tuple):
            y_hat = out[0]  # Extract sequence, ignore hidden state
        else:
            y_hat = out

    return y_hat.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    # -------------------------------------------------------------------------
    # 0. SETUP & GLOBAL CONFIGURATION
    # -------------------------------------------------------------------------
    global_config_path = Path("../global_config.yaml")

    if not global_config_path.exists():
        raise FileNotFoundError(f"Global config not found at {global_config_path}")

    with open(global_config_path) as f:
        global_config = yaml.safe_load(f)

    # Read target column from global schema (e.g., "glucose")
    target_col = global_config["schema"]["target_col"]

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 1. DATA PREPARATION (The "Chimera" Pipeline)
    # -------------------------------------------------------------------------
    print("\n>>> 1. Initializing and Preparing Datasets...")

    # Initialize datasets (Running INIT Phase: Mapping -> Indexing)
    ds1 = AZT1D2025Dataset(
        Path("../datasets/AZT1D2025/CGM Records"),
        Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        global_config_file=global_config_path,
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    ds2 = HUPA_UCMDataset(
        Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        global_config_file=global_config_path,
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )

    # ds3 = OhioT1DMDataset(
    #     Path("../datasets/OhioT1DMmini"),
    #     Path("../datasets/OhioT1DMmini/ohiot1dmmini.yaml"),
    #     global_config_file=global_config_path,
    #     logging_dir=Path("../datasets/OhioT1DMmini/prep_logs"),
    # )

    all_datasets = [ds1, ds2]

    # Execute the 3-Stage Pipeline for each dataset
    for ds in all_datasets:
        dataset_name = ds.config['dataset'].get('name', 'Unnamed')
        print(f"   Processing {dataset_name}...")

        # A. Clean: Repair types, remove duplicates, fill small gaps
        ds.clean()

        # B. Standardize: CRITICAL for mixed datasets.
        #    - Adds missing columns (filled with 0.0) based on global schema.
        #    - Creates '_mask' columns (1.0 if present, 0.0 if missing).
        ds.standardize()

        # C. Augment: Adds synthetic features (Time sin/cos, IOB/COB decay curves).
        ds.augment()

    # -------------------------------------------------------------------------
    # 2. DYNAMIC COLUMN DISCOVERY
    # -------------------------------------------------------------------------
    # Since augmentation adds new columns, we must inspect the dataframe
    # to find the final list of conditional features.
    # We take the first dataset as reference (standardize() ensures schema consistency).
    sample_df = ds1.all_data[0]

    # The conditional columns are ALL columns except the target
    final_cond_cols = [c for c in sample_df.columns if c != target_col]

    print(f"\n>>> Column Discovery Complete:")
    print(f"   Target Column: {target_col}")
    print(f"   Conditionals Detected ({len(final_cond_cols)}): {final_cond_cols}")
    # Example output: ['basal_rate', 'basal_rate_mask', 'tod_sin_24h', 'bolus_decay', ...]

    # -------------------------------------------------------------------------
    # 3. WINDOWING
    # -------------------------------------------------------------------------
    print("\n>>> 2. Building Windows...")

    cfg = ConditionalWindowingConfig(
        train_seq_len=24,  # Short length for quick testing (use 228 for production)
        train_step=12,
        val_seq_len=24,  # Short length for quick testing
        val_step=24,
        val_ratio=0.2,
        split_by="subject",
        random_state=42,
        batch_size=64,
        num_workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
        normalize=[target_col] + final_cond_cols,  # Normalize everything (including masks)
        freq_minutes=5,
    )

    builder = ConditionalWindowBuilder(cfg)

    # Build pack and loaders using the dynamically detected columns
    pack, train_loader, val_loader = builder.build_from_datasets(
        all_datasets,
        cond_cols=final_cond_cols,
        target_col=target_col,
    )

    # -------------------------------------------------------------------------
    # 4. TRAINING (TimeGAN)
    # -------------------------------------------------------------------------
    print(f"\n>>> 3. Training ConditionalTimeGAN...")
    print(f"   Input Dim: {len(final_cond_cols)} (Conditionals) -> Output Dim: 1 (Glucose)")

    model = ConditionalTimeGanModule(
        cond_dim=len(final_cond_cols),  # Input dimension matches detected features
        hidden_dim=32,  # Small hidden dim for testing
        num_layers=2,
        g_steps_per_iter=1,
    ).to(device)

    # Training Phases
    epochs = 2  # Few epochs for testing

    print("   [Phase 1] Autoencoder...")
    model.set_phase("ae")
    train_module(model, train_loader, val_loader, num_epochs=epochs, device=device)

    print("   [Phase 2] Supervisor...")
    model.set_phase("sup")
    train_module(model, train_loader, val_loader, num_epochs=epochs, device=device)

    print("   [Phase 3] Adversarial (Joint)...")
    model.set_phase("adv")
    train_module(model, train_loader, val_loader, num_epochs=epochs, device=device)

    # -------------------------------------------------------------------------
    # 5. GENERATION & RECONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n>>> 4. Generating & Reconstructing Validation Set...")

    # A. Generate synthetic windows (Normalized Tensors)
    y_hat_val = generate_val_target_windows(model, c_val=pack.c_val, device=device)

    # B. Reconstruct DataFrames (Denormalize and reassemble)
    recon = WindowReconstructor(
        ReconstructionConfig(
            target_col=pack.target_col,
            cond_cols=pack.cond_cols,
            synth_col="glucose_synth",
            include_true_target=True,  # Critical for evaluation (Real vs Synth comparison)
        )
    )

    synthetic_val_dfs = recon.reconstruct_subject_dfs(
        templates=pack.val_templates,
        meta=pack.meta_val,
        c_windows=pack.c_val,
        y_hat_windows=y_hat_val,
        scaling_params=pack.extra.get("scaling_params", None)
    )

    print(f"   Reconstructed {len(synthetic_val_dfs)} validation subject DataFrames.")

    # -------------------------------------------------------------------------
    # 6. SAVING & EVALUATION
    # -------------------------------------------------------------------------

    # A. SAVING (Safety Checkpoint)
    # Save the reconstructed dataframes so evaluation can be run separately
    output_dir = Path("../runs/timegan_chimera_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "reconstructed_val_data.pkl"
    print(f"\n>>> 5. Saving results to {save_path}...")

    save_results_as_csv_folder(
        dfs=synthetic_val_dfs,
        output_dir=output_dir,
        prefix="val_subject"
    )

    # Save the model state dict
    torch.save(model.state_dict(), output_dir / "timegan_model.pth")

    # B. EVALUATION (Placeholder)
    print("\n>>> 6. Running Evaluation Metrics...")

    try:
        # Future integration point for metrics pipeline
        # results = evaluate_dataset(synthetic_val_dfs, target_col="glucose", synth_col="glucose_synth")
        # print(results)
        print("   (Evaluation pipeline not yet attached - Data saved successfully)")
    except Exception as e:
        print(f"   [WARNING] Evaluation failed: {e}")
        print("   Don't worry, results are saved in .pkl for later analysis.")

    print("\n>>> Pipeline Completed Successfully. 🎉")


if __name__ == "__main__":
    main()