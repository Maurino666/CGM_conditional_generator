from __future__ import annotations

from pathlib import Path
import yaml
import torch
from datetime import datetime

# --- 1. Import Datasets ---
from data_prep import AZT1D2025Dataset, HUPA_UCMDataset

# --- 2. Import Data Manager Components ---
from data_management.splitter import DataSplitter
from data_management.normalization import MinMaxNormalizer

# --- 3. Import Windowing Components ---
from windowing import WindowBuilder, FullSequenceBuilder

# --- 4. Import Reconstruction Components ---
from reconstruction import ReconstructionConfig, FullSequenceReconstructor

# --- 5. Import Model (DIFFUSION) ---
# Ensure this imports the updated DiffWave module we discussed
from models import DiffWaveDiffusionModule

# --- 6. Import Training Components ---
from training import Trainer
from training.loggers import WandBLogger
from training.callbacks import GenerativeVisualizer

# --- 7. Import Inference Component ---
from inference import SequenceInferenceOrchestrator


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

    # --- Configuration Constants (Diffusion Specific) ---
    RUN_NAME = "diffwave_1"

    # Data Params
    SEQ_LEN = 288
    VAL_RATIO = 0.15

    # CNN-based models (DiffWave) usually consume less VRAM than RNNs,
    # so we might increase the batch size compared to TimeGAN.
    BATCH_SIZE = 384
    NUM_WORKERS = 4
    TRAIN_STEP = 12  # Stride for data augmentation
    SPLIT_STRATEGY = "subject"

    # Diffusion Model Architecture Params
    RESIDUAL_CHANNELS = 64
    NUM_LAYERS = 30
    CYCLE_LENGTH = 10  # Dilation cycle (1, 2, 4, 8, ...)
    TIMESTEPS = 1000  # T steps for the Forward/Reverse process
    LR = 2e-4  # Standard Learning Rate for DDPM

    # Training Params
    # Diffusion models typically require more epochs to converge than GANs.
    MAX_EPOCHS = 200
    VAL_CHECK_INTERVAL = 10  # Validate less frequently to save time

    # Config dict for WandB/Logging
    config = {
        "model_type": "DiffWave",
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "residual_channels": RESIDUAL_CHANNELS,
        "num_layers": NUM_LAYERS,
        "timesteps": TIMESTEPS,
        "lr": LR,
        "epochs": MAX_EPOCHS,
        "normalization": "minmax_neg1_pos1"
    }

    base_dir = Path("../runs/runs_diffusion")  # Separate directory for diffusion runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{RUN_NAME}"
    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------------------
    print("\n>>> 1. Loading Datasets...")

    # Initialize Datasets (Add/Remove as needed)
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

    # Feature Detection
    sample_df = ds1.all_data[0]

    # Combine ALL features into a single list
    all_feature_cols = [c for c in sample_df.columns if c != target_col]

    print(f"   Target: {target_col}")
    print(f"   Conditional Features: {len(all_feature_cols)} (Static + Dynamic)")

    # -------------------------------------------------------------------------
    # 2. SPLITTING
    # -------------------------------------------------------------------------
    print("\n>>> 2. Splitting Datasets...")
    splitter = DataSplitter(val_ratio=VAL_RATIO, random_state=42)
    train_dfs_raw, val_dfs_raw = splitter.split_data(datasets=all_datasets, strategy=SPLIT_STRATEGY)

    # -------------------------------------------------------------------------
    # 3. NORMALIZATION (CRITICAL: Range [-1, 1])
    # -------------------------------------------------------------------------
    print("\n>>> 3. Normalizing Data (Range -1 to 1)...")

    # IMPORTANT: Diffusion models rely on adding Gaussian Noise (mean=0, std=1).
    # To facilitate convergence, the input data should be symmetric around 0.
    # Therefore, we use a range of [-1, 1] instead of the standard [0, 1].
    normalizer = MinMaxNormalizer(
        cols_to_normalize=all_feature_cols + [target_col],
        feature_range=(-1, 1),
        fixed_ranges=global_config.get("normalization_ranges", None)
    )

    normalizer.fit(train_dfs_raw)
    normalizer.save_params(output_dir)
    train_dfs_norm = normalizer.transform(train_dfs_raw)
    val_dfs_norm = normalizer.transform(val_dfs_raw)

    # -------------------------------------------------------------------------
    # 4. WINDOWING (Strategy: Force all to Dynamic)
    # -------------------------------------------------------------------------
    print("\n>>> 4. Building Windows...")

    # CONFIGURATION TRICK:
    # We pass `static_cols=[]` (empty list).
    # This forces the WindowBuilder to treat static columns as dynamic ones.
    # Result: Static values are repeated for every time step in the sequence.
    # Benefit: The Loader will return exactly 2 tensors:
    #          1. Target [Batch, Seq_Len, 1]
    #          2. Condition [Batch, Seq_Len, N_Cond]
    # This perfectly matches the `_parse_batch` logic in our BaseDiffusionModule.

    builder = WindowBuilder(
        target_col=target_col,
        cond_cols=all_feature_cols,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        force_device=device
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
        step=SEQ_LEN,  # No overlap for validation
        shuffle=True,
        split_name="Validation"
    )

    # Runtime Check: Verify Loader Output Dimensions
    try:
        sample_batch = next(iter(train_split.loader))
        # sample_batch[0] is Target, sample_batch[1] is Condition
        sample_cond_dim = sample_batch[1].shape[-1]
        print(f"   [Check] Loader output dimensions confirmed.")
        print(f"   [Check] Condition Dimension: {sample_cond_dim}")
    except StopIteration:
        raise RuntimeError("Training loader is empty!")

    # -------------------------------------------------------------------------
    # 5. TRAINING (Single Loop)
    # -------------------------------------------------------------------------
    print(f"\n>>> Run info will be saved in: {output_dir}")
    print(f"\n>>> 5. Training Diffusion Model...")

    # A. Logger
    logger = WandBLogger(
        project_name="CGM_Diffusion_Gen",
        run_name=experiment_name,
        config=config,
        log_dir=output_dir,
    )

    # B. Model Instantiation
    model = DiffWaveDiffusionModule(
        input_dim=1,  # Target dimension (e.g., Glucose)
        cond_dim=sample_cond_dim,  # Calculated dynamically from loader
        residual_channels=RESIDUAL_CHANNELS,
        num_layers=NUM_LAYERS,
        cycle_length=CYCLE_LENGTH,
        timesteps=TIMESTEPS,
        lr=LR
    ).to(device)

    # C. Callbacks
    # We only use the Visualizer. PCA/Moment metrics can be computationally
    # expensive, so we might want to run them only at the end or less frequently.
    # Note: 'every_n_epochs' is set to 20 because DDPM generation is slow.
    callbacks = [
        GenerativeVisualizer(
            fixed_batch=next(iter(val_split.loader)),
            device=device,
            every_n_epochs=20
        )
    ]

    # D. Trainer Initialization
    # The generic Trainer handles the loops, while the Model handles the
    # specific DDPM logic (sampling t, adding noise, loss calc).
    trainer = Trainer(
        device=device,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=VAL_CHECK_INTERVAL
    )

    # E. Run Training
    trainer.fit(
        model=model,
        max_epochs=MAX_EPOCHS,
        train_loader=train_split.loader,
        val_loader=val_split.loader,
        callbacks=callbacks
    )

    print(f"\n   Saving final model to {output_dir}...")
    torch.save(model.state_dict(), output_dir / "diffwave_model.pth")
    logger.close()

    # -------------------------------------------------------------------------
    # 6. GENERATION & RECONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n>>> 6. Generation & Reconstruction...")

    # 1. Inference Builder
    gen_builder = FullSequenceBuilder(
        target_col=target_col,
        cond_cols=all_feature_cols,
        batch_size=1,
        num_workers=0,
        allow_target_nan=True
    )

    # 2. Reconstructor
    # Handles denormalization and dataframe creation.
    reconstructor = FullSequenceReconstructor(
        cfg=ReconstructionConfig(
            target_col=target_col,
            cond_cols=all_feature_cols,
            include_true_target=True
        ),
        normalizer=normalizer
    )

    # 3. Inference Orchestrator
    orchestrator = SequenceInferenceOrchestrator(
        model=model,
        builder=gen_builder,
        reconstructor=reconstructor,
        device=device,
        verbose=True
    )

    # 4. Generate Validation Set (Generalization Check)
    print("\n   [Generating Validation Data]...")
    orchestrator.run(
        dfs=val_dfs_norm,
        output_dir=output_dir / "val_gen",
        file_prefix="val_synthetic",
        split_name="Validation"
    )

    print(f"\n>>> Pipeline Completed Successfully. Results saved in {output_dir}")


if __name__ == "__main__":
    main()