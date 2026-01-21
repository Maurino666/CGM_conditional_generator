from __future__ import annotations

from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. Import Datasets ---
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset

# --- 2. Import New Data Manager Components ---
from data_management.splitter import DataSplitter
from data_management.normalization import MinMaxNormalizer

# --- 3. Import New Windowing Components ---
from windowing import WindowBuilder, ConditionalWindowPack, WindowSplit

# --- 4. Import Reconstruction Components ---
from reconstruction import WindowReconstructor, ReconstructionConfig

# --- 5. Import Model ---
# Assuming you kept the model class in the 'models' package or moved it to 'src.models'
# Adjust this import based on your current file structure.
from models import ConditionalTimeGanModule

# --- 6. Import NEW Training Components (The Refactor) ---
from src.training.trainer import Trainer
from src.training.loggers import TensorBoardLogger
from src.training.callbacks.visualization import GenerativeVisualizer


# [Helper functions remain unchanged]
def save_results_as_csv_folder(dfs: list[pd.DataFrame], output_dir: Path, prefix: str = "synthetic_subject") -> None:
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
            if subject_id:
                if data_source:
                    filename = f"{prefix}_{str(data_source)}_{subject_id}.csv"
                else:
                    filename = f"{prefix}_{str(subject_id)}.csv"
            else:
                filename = f"{prefix}_{i:03d}.csv"
        save_path = csv_dir / filename
        df.to_csv(save_path, index=True)
    print("   CSV saving complete.")


def generate_val_target_windows(model: torch.nn.Module, c_val: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    c = torch.as_tensor(c_val, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model.generate(c)
        if isinstance(out, tuple):
            y_hat = out[0]
        else:
            y_hat = out
    return y_hat.detach().cpu().numpy().astype(np.float32)


def process_split_generation(model: torch.nn.Module, split: WindowSplit, reconstructor: WindowReconstructor,
                             scaling_params: dict[str, tuple[float, float]], device: torch.device, split_name: str) -> \
list[pd.DataFrame]:
    print(f"\n   [Processing Generation: {split_name.upper()}]")
    if len(split) == 0:
        print("   [!] Warning: Split is empty. Skipping.")
        return []
    print(f"   Generating synthetic data for {len(split)} windows...")
    y_hat = generate_val_target_windows(model, split.c, device=device)
    print(f"   Reconstructing into DataFrames...")
    dfs = reconstructor.reconstruct(templates=split.templates, meta=split.metadata, y_hat_windows=y_hat,
                                    scaling_params=scaling_params)
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
    SEQ_LEN = 36
    VAL_RATIO = 0.2
    BATCH_SIZE = 128
    TRAIN_STEP = 12
    SPLIT_STRATEGY = "subject"

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION (Load & Clean)
    # -------------------------------------------------------------------------
    print("\n>>> 1. Loading Datasets...")
    # [Data loading code remains identical to previous version]
    ds1 = AZT1D2025Dataset(
        dataset_root=Path("../datasets/AZT1D2025/CGM Records"),
        config_file=Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        global_config_file=global_config_path,
        patient_metadata_path=Path("../datasets/AZT1D2025/CGM Records/patient_metadata.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    ds2 = HUPA_UCMDataset(
        dataset_root=Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        config_file=Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        global_config_file=global_config_path,
        patient_metadata_path=Path("../datasets/HUPA-UCM Diabetes Dataset/patient_metadata.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )

    all_datasets = [ds1, ds2]
    for ds in all_datasets:
        print(f"   Processing {ds.config['dataset'].get('name')}...")
        ds.clean()
        ds.standardize()
        ds.augment()

    sample_df = ds1.all_data[0]
    final_cond_cols = [c for c in sample_df.columns if c != target_col]
    print(f"   Features Detected: {len(final_cond_cols)} conditional columns.")

    # -------------------------------------------------------------------------
    # 2. SPLITTING
    # -------------------------------------------------------------------------
    print("\n>>> 2. Splitting Datasets...")
    splitter = DataSplitter(val_ratio=VAL_RATIO, random_state=42)
    train_dfs_raw, val_dfs_raw = splitter.split_data(datasets=all_datasets, strategy=SPLIT_STRATEGY)

    # -------------------------------------------------------------------------
    # 3. NORMALIZATION
    # -------------------------------------------------------------------------
    print("\n>>> 3. Normalizing Data...")
    normalizer = MinMaxNormalizer(
        cols_to_normalize=final_cond_cols + [target_col],
        fixed_ranges=global_config.get("normalization_ranges", None)
    )
    normalizer.fit(train_dfs_raw)
    train_dfs_norm = normalizer.transform(train_dfs_raw)
    val_dfs_norm = normalizer.transform(val_dfs_raw)
    print(f"Normalization parameters: {normalizer.get_params()}")

    # -------------------------------------------------------------------------
    # 4. WINDOWING
    # -------------------------------------------------------------------------
    print("\n>>> 4. Building Windows...")
    builder = WindowBuilder(target_col=target_col, cond_cols=final_cond_cols, batch_size=BATCH_SIZE, num_workers=4)

    train_split_optim = builder.build_subset(dfs=train_dfs_norm, seq_len=SEQ_LEN, step=TRAIN_STEP, shuffle=True,
                                             split_name="Train_Optimized")
    val_split = builder.build_subset(dfs=val_dfs_norm, seq_len=SEQ_LEN, step=SEQ_LEN, shuffle=False,
                                     split_name="Validation")

    pack = ConditionalWindowPack(
        train_split=train_split_optim,
        val_split=val_split,
        target_col=target_col,
        cond_cols=final_cond_cols,
        scaling_params=normalizer.get_params()
    )

    # -------------------------------------------------------------------------
    # 5. TRAINING (MODIFIED WITH NEW ARCHITECTURE)
    # -------------------------------------------------------------------------
    base_dir = Path("../runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_TimeGAN_Modular_Test"
    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Run info will be saved in: {output_dir}")
    print(f"\n>>> 5. Training ConditionalTimeGAN (Modular Trainer)...")

    # A. Setup Logger
    # We initialize the TensorBoard logger pointing to the experiment directory
    tb_logger = TensorBoardLogger(log_dir=output_dir / "tensorboard")

    # B. Setup Visualizer Callback
    # We grab a single batch from the validation loader to serve as a fixed reference
    # for generating comparison plots throughout training.
    try:
        fixed_vis_batch = next(iter(pack.val_split.loader))
        print("   [Setup] Fixed validation batch acquired for visualization.")
    except StopIteration:
        print("   [Setup] WARNING: Validation loader is empty! Visualization disabled.")
        fixed_vis_batch = None

    visualizer = GenerativeVisualizer(
        fixed_batch=fixed_vis_batch,
        device=device,
        every_n_epochs=1  # Generate plot every 5 epochs
    )

    # C. Instantiate Model
    model = ConditionalTimeGanModule(
        cond_dim=len(final_cond_cols),
        hidden_dim=15,
        num_layers=1,
        g_steps_per_iter=1,
    ).to(device)

    # --- Phase 1: Autoencoder ---
    print("\n   [Phase 1] Autoencoder...")
    model.set_phase("ae")

    # We create a specific trainer for this phase.
    # Note: We don't use the visualizer here because AE reconstruction
    # is usually tracked via MSE loss, not generation.
    trainer = Trainer(
        device=device,
        logger=tb_logger,
    )
    # Fit the model (using train and val loaders)
    model.set_phase("ae")
    trainer.fit(model, 10 ,pack.train_split.loader, pack.val_split.loader)

    # --- Phase 2: Supervisor ---
    print("\n   [Phase 2] Supervisor...")
    model.set_phase("sup")
    trainer.fit(model, 5,pack.train_split.loader, pack.val_split.loader)

    # --- Phase 3: Adversarial (Joint) ---
    print("\n   [Phase 3] Adversarial (Joint)...")
    model.set_phase("adv")

    # In this phase, we attach the visualizer callback to see
    # if the generator learns to produce realistic data.
    trainer.fit(model, 2, pack.train_split.loader, pack.val_split.loader, callbacks=[visualizer])

    # Manual Save (Temporary, as requested without Checkpoint Callback)
    print(f"\n   Saving final model to {output_dir}...")
    torch.save(model.state_dict(), output_dir / "timegan_model.pth")

    # Close logger explicitly
    tb_logger.close()

    # -------------------------------------------------------------------------
    # 6. GENERATION & RECONSTRUCTION (Standard Pipeline)
    # -------------------------------------------------------------------------
    print("\n>>> 6. Generation & Reconstruction...")

    builder = WindowBuilder(
        target_col=target_col,
        cond_cols=final_cond_cols,
        batch_size=BATCH_SIZE,
        num_workers=4,
        max_missing_ratio=0.05,
        allow_target_nan=True,
    )

    val_gen_split = builder.build_subset(
        dfs=val_dfs_norm,
        seq_len=SEQ_LEN,
        step=SEQ_LEN,
        shuffle=False,
        split_name="Val_generation"
    )

    reconstructor = WindowReconstructor(
        cfg=ReconstructionConfig(target_col=target_col, cond_cols=final_cond_cols, include_true_target=True),
        strategy="overwrite"
    )

    synth_val_dfs = process_split_generation(model, val_gen_split, reconstructor, pack.scaling_params, device,
                                             "Validation")

    print("\n   [TSTR Prep] Re-building Training Windows (Non-Overlapping)...")
    train_split_tstr = builder.build_subset(
        dfs=train_dfs_norm,
        seq_len=SEQ_LEN,
        step=SEQ_LEN,
        shuffle=False,
        split_name="Train_TSTR_Subset"
    )

    synth_train_dfs = process_split_generation(model, train_split_tstr, reconstructor, pack.scaling_params, device,
                                               "Train_TSTR_Synth")

    # -------------------------------------------------------------------------
    # 7. SAVING
    # -------------------------------------------------------------------------
    print(f"\n>>> 7. Saving Results to {output_dir}...")

    save_results_as_csv_folder(synth_val_dfs, output_dir / "val", prefix="val_subject")
    save_results_as_csv_folder(synth_train_dfs, output_dir/ "train", prefix="train_tstr_subject")

    print("\n>>> Pipeline Completed Successfully. 🎉")


if __name__ == "__main__":
    main()