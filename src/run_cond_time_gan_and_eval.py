from __future__ import annotations

from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd

# --- 1. Import Datasets ---
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset

# --- 2. Import New Data Manager Components ---
from data_management.splitter import DataSplitter
from data_management.normalization import MinMaxNormalizer

# --- 3. Import New Windowing Components ---
from windowing import WindowBuilder, ConditionalWindowPack, WindowSplit

# --- 4. Import Reconstruction Components ---
from reconstruction import WindowReconstructor, ReconstructionConfig

# --- 5. Import Model and Trainer ---
from models import ConditionalTimeGanModule, train_module


def save_results_as_csv_folder(
        dfs: list[pd.DataFrame],
        output_dir: Path,
        prefix: str = "synthetic_subject"
) -> None:
    """
    Saves a list of DataFrames as individual CSV files.
    """
    csv_dir = output_dir / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Saving {len(dfs)} CSV files to: {csv_dir}")



    for i, df in enumerate(dfs):
        unique_id = df.attrs.get("unique_id", None)

        if unique_id:
            filename = f"{prefix}_{str(unique_id)}.csv"
        else:
            subject_id = df.attrs.get("subject_id", None)
            data_source = df.attrs.get("dataset_source", None)

            if  subject_id:
                if data_source:
                    filename = f"{prefix}_{str(data_source)}_{subject_id}.csv"
                else:
                    filename = f"{prefix}_{str(subject_id)}.csv"
            else:
                filename = f"{prefix}_{i:03d}.csv"


        save_path = csv_dir / filename
        df.to_csv(save_path, index=True)

    print("   CSV saving complete.")


def generate_val_target_windows(
        model: torch.nn.Module,
        c_val: np.ndarray,
        device: torch.device,
) -> np.ndarray:
    """
    Runs model inference to generate synthetic target windows.
    """
    model.eval()
    c = torch.as_tensor(c_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model.generate(c)
        if isinstance(out, tuple):
            y_hat = out[0]
        else:
            y_hat = out

    return y_hat.detach().cpu().numpy().astype(np.float32)


def process_split_generation(
        model: torch.nn.Module,
        split: WindowSplit,
        reconstructor: WindowReconstructor,
        scaling_params: dict[str, tuple[float, float]],
        device: torch.device,
        split_name: str
) -> list[pd.DataFrame]:
    """
    Helper function to:
    1. Take a specific WindowSplit (Train or Val).
    2. Generate synthetic data using the Model.
    3. Reconstruct the data into DataFrames.
    """
    print(f"\n   [Processing Generation: {split_name.upper()}]")

    if len(split) == 0:
        print("   [!] Warning: Split is empty. Skipping.")
        return []

    # 1. Generate Synthetic Targets
    print(f"   Generating synthetic data for {len(split)} windows...")
    y_hat = generate_val_target_windows(model, split.c, device=device)

    # 2. Reconstruct
    # We pass the metadata and templates contained within the Split object
    print(f"   Reconstructing into DataFrames...")
    dfs = reconstructor.reconstruct(
        templates=split.templates,
        meta=split.metadata,
        y_hat_windows=y_hat,
        scaling_params=scaling_params
    )

    return dfs


def main() -> None:
    # -------------------------------------------------------------------------
    # 0. SETUP & GLOBAL CONFIGURATION
    # -------------------------------------------------------------------------
    global_config_path = Path("../global_config.yaml")

    if not global_config_path.exists():
        raise FileNotFoundError(f"Global config not found at {global_config_path}")

    with open(global_config_path) as f:
        global_config = yaml.safe_load(f)

    target_col = global_config["schema"]["target_col"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Configuration Constants
    SEQ_LEN = 24
    VAL_RATIO = 0.2
    BATCH_SIZE = 2048
    SPLIT_STRATEGY = "subject"  # 'subject' (Stratified) or 'time'

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION (Load & Clean)
    # -------------------------------------------------------------------------
    print("\n>>> 1. Loading Datasets...")

    ds1 = AZT1D2025Dataset(
        dataset_root=Path("../datasets/AZT1D2025/CGM Records"),
        config_file=Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        global_config_file=global_config_path,
        patient_metadata_path=Path("../datasets/AZT1D2025/CGM Records/patient_metadata.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    ds2 = HUPA_UCMDataset(
        dataset_root= Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        config_file= Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        global_config_file=global_config_path,
        patient_metadata_path= Path("../datasets/HUPA-UCM Diabetes Dataset/patient_metadata.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )

    all_datasets = [ds1, ds2]

    # Clean, Standardize, Augment
    for ds in all_datasets:
        print(f"   Processing {ds.config['dataset'].get('name')}...")
        ds.clean()
        ds.standardize()
        ds.augment()

    # Dynamic Column Discovery (from first dataset)
    sample_df = ds1.all_data[0]
    final_cond_cols = [c for c in sample_df.columns if c != target_col]
    print(f"   Features Detected: {len(final_cond_cols)} conditional columns.")

    # -------------------------------------------------------------------------
    # 2. SPLITTING (Explicit & Stratified)
    # -------------------------------------------------------------------------
    print("\n>>> 2. Splitting Datasets...")

    splitter = DataSplitter(val_ratio=VAL_RATIO, random_state=42)

    # Returns pure lists of DataFrames (with injected IDs in df.attrs)
    train_dfs_raw, val_dfs_raw = splitter.split_data(
        datasets=all_datasets,
        strategy=SPLIT_STRATEGY
    )



    # -------------------------------------------------------------------------
    # 3. NORMALIZATION (Fit on Train -> Apply to All)
    # -------------------------------------------------------------------------
    print("\n>>> 3. Normalizing Data...")

    normalizer = MinMaxNormalizer(
        cols_to_normalize=final_cond_cols + [target_col],
        fixed_ranges=global_config.get("normalization_ranges", None)
    )

    # A. Fit only on Training data (Prevents Leakage)
    normalizer.fit(train_dfs_raw)

    # B. Transform both sets
    train_dfs_norm = normalizer.transform(train_dfs_raw)
    val_dfs_norm = normalizer.transform(val_dfs_raw)

    # -------------------------------------------------------------------------
    # 4. WINDOWING (The Agnostic Builder)
    # -------------------------------------------------------------------------
    print("\n>>> 4. Building Windows...")

    builder = WindowBuilder(
        target_col=target_col,
        cond_cols=final_cond_cols,
        batch_size=BATCH_SIZE,
        num_workers= 4,
        max_missing_ratio=0.05
    )

    # A. TRAIN SPLIT (Optimized for Learning)
    # High overlap (step=1) to maximize training examples.
    train_split_optim = builder.build_subset(
        dfs=train_dfs_norm,
        seq_len=SEQ_LEN,
        step=1,  # Maximum Overlap
        shuffle=True,  # Shuffle for training
        split_name="Train_Optimized"
    )

    # B. VAL SPLIT (Optimized for Evaluation)
    # No overlap (step=SEQ_LEN) or minimal overlap.
    val_split = builder.build_subset(
        dfs=val_dfs_norm,
        seq_len=SEQ_LEN,
        step=SEQ_LEN,  # Non-Overlapping
        shuffle=False,  # Keep order for reconstruction
        split_name="Validation"
    )

    # C. CREATE PACK (For Trainer)
    # Links the optimized training data and validation data
    pack = ConditionalWindowPack(
        train_split=train_split_optim,
        val_split=val_split,
        target_col=target_col,
        cond_cols=final_cond_cols,
        scaling_params=normalizer.get_params()
    )

    # -------------------------------------------------------------------------
    # 5. TRAINING
    # -------------------------------------------------------------------------

    output_dir = Path("../runs/full_pipeline_true_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> 5. Training ConditionalTimeGAN...")

    model = ConditionalTimeGanModule(
        cond_dim=len(final_cond_cols),
        hidden_dim=256,
        num_layers=3,
        g_steps_per_iter=1,
    ).to(device)



    print("   [Phase 1] Autoencoder...")
    model.set_phase("ae")
    train_module(
        model,
        pack.train_split.loader,
        pack.val_split.loader,
        10,
        device,
        tensorboard_dir=output_dir / "tensorboard",
    )

    print("   [Phase 2] Supervisor...")
    model.set_phase("sup")
    train_module(
        model,
        pack.train_split.loader,
        pack.val_split.loader,
        5,
        device,
        tensorboard_dir=output_dir / "tensorboard",
    )

    print("   [Phase 3] Adversarial (Joint)...")
    model.set_phase("adv")
    train_module(
        model,
        pack.train_split.loader,
        pack.val_split.loader,
        100,
        device,
        tensorboard_dir=output_dir / "tensorboard",
    )

    # Save Model
    torch.save(model.state_dict(), output_dir / "timegan_model.pth")

    # -------------------------------------------------------------------------
    # 6. GENERATION & RECONSTRUCTION (Advanced)
    # -------------------------------------------------------------------------
    print("\n>>> 6. Generation & Reconstruction...")

    # We use 'overwrite' strategy because we will generate non-overlapping data
    reconstructor = WindowReconstructor(
        cfg=ReconstructionConfig(
            target_col=target_col,
            cond_cols=final_cond_cols,
            include_true_target=True
        ),
        strategy="overwrite"
    )

    # A. VALIDATION GENERATION (Standard Inference)
    # Generates synthetic data for the held-out patients (Patient E)
    synth_val_dfs = process_split_generation(
        model, val_split, reconstructor,
        pack.scaling_params, device, "Validation"
    )

    # B. TSTR GENERATION (Train on Synthetic, Test on Real)
    # We need to generate a synthetic version of the TRAINING set.
    # CRITICAL: We create a NEW split with NO OVERLAP to avoid "flattening" metrics.

    print("\n   [TSTR Prep] Re-building Training Windows (Non-Overlapping)...")
    train_split_tstr = builder.build_subset(
        dfs=train_dfs_norm,
        seq_len=SEQ_LEN,
        step=SEQ_LEN,  # <--- STRIDE = SEQ_LEN (No Overlap)
        shuffle=False,  # Keep order
        split_name="Train_TSTR_Subset"
    )

    synth_train_dfs = process_split_generation(
        model, train_split_tstr, reconstructor,
        pack.scaling_params, device, "Train_TSTR_Synth"
    )

    # -------------------------------------------------------------------------
    # 7. SAVING
    # -------------------------------------------------------------------------
    output_dir = Path("../runs/full_pipeline_true_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> 7. Saving Results to {output_dir}...")

    # Split dfs raw to debug
    save_results_as_csv_folder(train_dfs_raw, output_dir / "split" / "train", prefix="train")
    save_results_as_csv_folder(val_dfs_raw, output_dir / "split" / "val", prefix="val")


    # Save Validation (Test) Results
    save_results_as_csv_folder(synth_val_dfs, output_dir / "val", prefix="val_subject")

    # Save TSTR Training Results
    save_results_as_csv_folder(synth_train_dfs, output_dir / "train", prefix="train_tstr_subject")


    print("\n>>> Pipeline Completed Successfully. 🎉")
    print("    You now have:")
    print("    1. 'val_subject_*.csv': Synthetic data for new patients (Generalization).")
    print("    2. 'train_tstr_subject_*.csv': Synthetic data replacing real training data (Utility).")


if __name__ == "__main__":
    main()