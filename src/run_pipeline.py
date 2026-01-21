from __future__ import annotations

from pathlib import Path
import yaml
import torch
from datetime import datetime

# --- 1. Import Datasets ---
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset

# --- 2. Import New Data Manager Components ---
from data_management.splitter import DataSplitter
from data_management.normalization import MinMaxNormalizer

# --- 3. Import New Windowing Components ---
from windowing import WindowBuilder, ConditionalWindowPack

# --- 4. Import Reconstruction Components ---
from reconstruction import WindowReconstructor, ReconstructionConfig

# --- 5. Import Model ---
from models import ConditionalTimeGanModule

# --- 6. Import NEW Training Components ---
from training import Trainer
from training.loggers import TensorBoardLogger
from training.callbacks import GenerativeVisualizer

# --- 7. Import Inference Component ---
from inference import InferenceOrchestrator


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
    SEQ_LEN = 288
    VAL_RATIO = 0.2
    BATCH_SIZE = 384
    NUM_WORKERS = 4
    TRAIN_STEP = 12
    SPLIT_STRATEGY = "subject"
    # TimeGan params
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    G_STEPS_PER_ITER = 2
    NOISE_STD = 0.1
    SOFT_LABEL = 0.9
    # Training params
    AE_EPOCHS = 10
    SUP_EPOCHS = 10
    ADV_EPOCHS = 100
    # Builder extras
    FORCE_DEVICE = device

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION (Load & Clean)
    # -------------------------------------------------------------------------
    print("\n>>> 1. Loading Datasets...")
    ds1 = AZT1D2025Dataset(
        dataset_root=Path("../datasets/AZT1D2025/CGM Records"),
        config_file=Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        global_config_file=global_config_path,
        # patient_metadata_path=Path("../datasets/AZT1D2025/CGM Records/patient_metadata.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    ds2 = HUPA_UCMDataset(
        dataset_root=Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
        config_file=Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
        global_config_file=global_config_path,
        # patient_metadata_path=Path("../datasets/HUPA-UCM Diabetes Dataset/patient_metadata.yaml"),
        logging_dir=Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
    )

    all_datasets = [ds1, ds2]
    for ds in all_datasets:
        print(f"   Processing {ds.config['dataset'].get('name')}...")
        ds.clean()
        ds.standardize()
        # ds.augment()

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
    builder = WindowBuilder(
        target_col=target_col,
        cond_cols=final_cond_cols,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        force_device=FORCE_DEVICE,
    )

    train_split = builder.build_subset(
        dfs=train_dfs_norm,
        seq_len=SEQ_LEN,
        step=TRAIN_STEP,
        shuffle=True,
        split_name="Train"
    )
    val_split = builder.build_subset(
        dfs=val_dfs_norm,
        seq_len=SEQ_LEN,
        step=SEQ_LEN,
        shuffle=False,
        split_name="Validation"
    )

    pack = ConditionalWindowPack(
        train_split=train_split,
        val_split=val_split,
        target_col=target_col,
        cond_cols=final_cond_cols,
        scaling_params=normalizer.get_params()
    )

    # -------------------------------------------------------------------------
    # 5. TRAINING
    # -------------------------------------------------------------------------
    base_dir = Path("../runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_CondTimeGan_No_Static_No_augment"
    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Run info will be saved in: {output_dir}")
    print(f"\n>>> 5. Training ConditionalTimeGAN (Modular Trainer)...")

    # A. Setup Logger
    tb_logger = TensorBoardLogger(log_dir=output_dir / "tensorboard")

    # B. Setup Visualizer Callback
    try:
        fixed_vis_batch = next(iter(pack.val_split.loader))
        print("   [Setup] Fixed validation batch acquired for visualization.")
    except StopIteration:
        print("   [Setup] WARNING: Validation loader is empty! Visualization disabled.")
        fixed_vis_batch = None

    visualizer = GenerativeVisualizer(
        fixed_batch=fixed_vis_batch,
        device=device,
        every_n_epochs=5
    )

    # C. Instantiate Model
    model = ConditionalTimeGanModule(
        cond_dim=len(final_cond_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        g_steps_per_iter=G_STEPS_PER_ITER,
        noise_std=NOISE_STD,
        soft_label=SOFT_LABEL,
    ).to(device)

    # --- Phase 1: Autoencoder ---
    print("\n   [Phase 1] Autoencoder...")
    model.set_phase("ae")
    trainer = Trainer(device=device, logger=tb_logger, log_every_n_steps= 50)
    trainer.fit(model, AE_EPOCHS, pack.train_split.loader, pack.val_split.loader)

    # --- Phase 2: Supervisor ---
    print("\n   [Phase 2] Supervisor...")
    model.set_phase("sup")
    trainer.fit(model, SUP_EPOCHS, pack.train_split.loader, pack.val_split.loader)

    # --- Phase 3: Adversarial (Joint) ---
    print("\n   [Phase 3] Adversarial (Joint)...")
    model.set_phase("adv")
    trainer.fit(model, ADV_EPOCHS, pack.train_split.loader, pack.val_split.loader, callbacks=[visualizer])

    print(f"\n   Saving final model to {output_dir}...")
    torch.save(model.state_dict(), output_dir / "timegan_model.pth")
    tb_logger.close()

    # -------------------------------------------------------------------------
    # 6. GENERATION & RECONSTRUCTION (New Inference Orchestrator)
    # -------------------------------------------------------------------------
    print("\n>>> 6. Generation & Reconstruction...")

    # 1. Setup Inference Components
    # We create a specific WindowBuilder for generation (inference mode).
    # 'allow_target_nan=True' allows generating even if target data has gaps (common in real data).
    gen_builder = WindowBuilder(
        target_col=target_col,
        cond_cols=final_cond_cols,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_target_nan=True,
    )

    # Setup Reconstructor
    # 'include_true_target=True' preserves the original target column in the output DF
    # for direct comparison/evaluation.
    reconstructor = WindowReconstructor(
        cfg=ReconstructionConfig(
            target_col=target_col,
            cond_cols=final_cond_cols,
            include_true_target=True
        ),
        strategy="overwrite"
    )

    # 2. Instantiate Orchestrator
    orchestrator = InferenceOrchestrator(
        model=model,
        window_builder=gen_builder,
        reconstructor=reconstructor,
        device=device,
        verbose=True
    )

    # 3. Generate Validation Split (Generalization Test)
    # Generates: output_dir/val/csv_data/val_subject_X.csv
    print("\n   [Generating Validation Data]...")
    orchestrator.run(
        dfs=val_dfs_norm,
        seq_len=SEQ_LEN,
        output_dir=output_dir / "val",
        file_prefix="val_subject",
        scaling_params=pack.scaling_params,
        split_name="Validation"
    )

    # 4. Generate Training Split (TSTR Utility)
    # Generates: output_dir/train/csv_data/train_tstr_subject_X.csv
    # This creates a synthetic version of the training set for Train-on-Synthetic-Test-on-Real evaluation.
    print("\n   [Generating TSTR Training Data]...")
    orchestrator.run(
        dfs=train_dfs_norm,
        seq_len=SEQ_LEN,
        output_dir=output_dir / "train",
        file_prefix="train_tstr_subject",
        scaling_params=pack.scaling_params,
        split_name="Train_TSTR"
    )

    print(f"\n>>> Pipeline Completed Successfully. Results saved in {output_dir}")


if __name__ == "__main__":
    main()