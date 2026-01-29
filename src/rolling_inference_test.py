import torch
import pandas as pd
from pathlib import Path
import logging

from data_management.normalization import MinMaxNormalizer
from inference.rolling import RollingInferenceOrchestrator
from models import ProjectedStaticTimeGanModule, StaticConditionalTimeGanModule
from reconstruction import WindowReconstructor, ReconstructionConfig
from windowing import WindowBuilder


RUN_NAME = "20260122_145150_Fixed_Static_SW1_MW1"
INPUT_DATA_DIR = Path(f"../runs/{RUN_NAME}/val/csv_data")
OUTPUT_DIR = Path(f"../runs/{RUN_NAME}/val_rolling")
MODEL_WEIGHTS = Path(f"../runs/{RUN_NAME}/timegan_model.pth")
NORMALIZATION_PARAMS_FILE = Path(f"../runs/{RUN_NAME}/normalization_params.json")

# Configuration Constants
SEQ_LEN = 288
VAL_RATIO = 0.15
BATCH_SIZE = 512
NUM_WORKERS = 4
TRAIN_STEP = 12
SPLIT_STRATEGY = "subject"
# TimeGan params
HIDDEN_DIM = 128
NUM_LAYERS = 2
NOISE_DIM = 64
G_STEPS_PER_ITER = 3
NOISE_STD = 0.1
SOFT_LABEL = 0.9
SUPERVISED_WEIGHT = 5.0
MOMENT_WEIGHT = 1.0
GAMMA = 1.0
LR = 1e-4
D_LOSS_THRESHOLD = 0.5


TARGET_COL = "glucose"
DYNAMIC_COLS = [
    "basal_rate",
    "heart_rate",
    "steps",
    "calories_burned",
    "bolus_total",
    "bolus_correction",
    "bolus_meal",
    "carbs",
    "basal_rate_mask",
    "heart_rate_mask",
    "steps_mask",
    "calories_burned_mask",
    "bolus_total_mask",
    "bolus_correction_mask",
    "bolus_meal_mask",
    "carbs_mask",
    "tod_sin_24h",
    "tod_cos_24h",
]
STATIC_COLS = ["age", "sex_m", "hba1c", "age_mask", "sex_m_mask", "hba1c_mask"]

ALL_COLS = STATIC_COLS + DYNAMIC_COLS

STATIC_REFRESH_RATE = 0.3

TIME_COL = "timestamp"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Function to load data from directory
def load_data(data_dir: Path) -> list[pd.DataFrame]:
    """Loads all csv files in a directory, to a list of dataframes"""
    files = list(data_dir.glob("*.csv"))
    dfs = []
    logging.info(f"Found {len(files)} CSV files in {data_dir}")
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=[TIME_COL], index_col=TIME_COL)
            df.attrs['subject_id'] = f.stem
            dfs.append(df)
        except Exception as e:
            logging.error(f"Loading Error {f}: {e}")
    return dfs


def main():
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # loading data
    test_dfs = load_data(INPUT_DATA_DIR)

    # initializing model
    model = StaticConditionalTimeGanModule(
        cond_dim=len(DYNAMIC_COLS),
        static_dim=len(STATIC_COLS),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        noise_dim=NOISE_DIM,
        g_steps_per_iter=G_STEPS_PER_ITER,
        noise_std=NOISE_STD,
        soft_label=SOFT_LABEL,
        supervised_weight=SUPERVISED_WEIGHT,
        moment_weight=MOMENT_WEIGHT,
        gamma=GAMMA,
        lr=LR,
        d_loss_threshold=D_LOSS_THRESHOLD,
    )

    # loading model weights
    try:
        state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logging.info("Weights loaded correctly.")
    except FileNotFoundError:
        logging.error(f"File not found: {MODEL_WEIGHTS}")
        return

    # normalization df normalization
    normalizer = MinMaxNormalizer(
        ALL_COLS + [TARGET_COL],
    )

    normalizer.load_params(NORMALIZATION_PARAMS_FILE)

    test_dfs = normalizer.transform(test_dfs)

    builder = WindowBuilder(
        target_col=TARGET_COL,
        cond_cols=STATIC_COLS + DYNAMIC_COLS,
        static_cols=STATIC_COLS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,

        allow_target_nan=True,
    )

    reconstructor = WindowReconstructor(
        cfg= ReconstructionConfig(
            target_col=TARGET_COL,
            cond_cols=STATIC_COLS+DYNAMIC_COLS,

        ),
        normalizer=normalizer,
    )

    orchestrator = RollingInferenceOrchestrator(
        model=model,
        window_builder=builder,
        reconstructor=reconstructor,
        device=DEVICE,
        seq_len=SEQ_LEN,
        verbose=True
    )

    logging.info("Starting rolling generation...")

    generated_dfs = orchestrator.run(
        dfs=test_dfs,
        static_refresh_rate=STATIC_REFRESH_RATE,
        output_dir=OUTPUT_DIR,
        file_prefix="Synth_Rolling",
        split_name="Test_Set_Rolling"
    )

    logging.info(f"Generation completed! {len(generated_dfs)} files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()