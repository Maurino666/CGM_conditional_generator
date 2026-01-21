# Generative Pipeline for Conditional CGM Synthesis

This project implements a modular **generative pipeline** designed to experiment with deep learning models for the conditional generation of Continuous Glucose Monitoring (CGM) data.

The system is built to ingest clinical Type 1 Diabetes (T1DM) datasets, normalize and window the data, and train generative models to produce realistic synthetic glucose traces conditioned on many situational variables. The pipeline is structured to support comparative analysis across different model configurations.

## 📂 Supported Datasets

The pipeline currently supports data ingestion and harmonization from:
1.  **HUPA-UCM Diabetes Dataset**: Clinical data from 25 patients (Open Access).
2.  **AZT1D Dataset**: Real-world data from patients using Automated Insulin Delivery (AID) systems.

> **Note:** Raw datasets are not included in this repository.

## 🚀 Installation on Linux

Follow these steps to set up the environment on a generic Linux machine with NVIDIA GPU support.

### 1. Clone the Repository
```bash
git clone https://github.com/Maurino666/CGM_conditional_generator/
cd CGM_conditional_generator
```

### 2. Set up the Virtual Environment
Using a virtual environment is highly recommended to isolate dependencies.

```bash
# Create the environment named 'venv'
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

### 3. Install PyTorch (Manual Step)
We install PyTorch separately to ensure the correct Linux/CUDA drivers are targeted.

```bash
# This command installs PyTorch with the latest stable CUDA support
pip install torch torchvision torchaudio
```

### 4. Install Dependencies
Install the remaining data science and utility packages.

```bash
pip install -r requirements.txt
```

### 5. Verify GPU Access
Run this quick check to ensure the code can see your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚙️ Configuration

The pipeline is data-driven and configured via **`global_config.yaml`**.
You can modify this file to change:
* **Schema Definitions**: Target columns (e.g., `glucose`) and conditional features (`carbs`, `basal_rate`, ...).
* **Normalization**: Custom ranges for MinMax scaling.

Model hyperparameters are currently set in the entry point scripts.

## 📁 Project Modules

* **`data_prep/`**: Extendable drivers for ingesting specific datasets (AZT1D, HUPA) and handling unit inconsistencies (e.g., adaptive carb scaling).
* **`data_management/`**: Core logic for splitting (Train/Val) and normalization.
* **`windowing/`**: Agnostic window builder that converts DataFrames into sliding window tensors for RNNs/GANs.
* **`models/`**: PyTorch implementations of generative models.
* **`reconstruction/`**: Inverse transformation logic to convert synthetic tensors back into analyzeable DataFrames.
* **`evaluation/`**: Customizable evaluation pipeline, which already contains some core metrics.

## 🛠 Troubleshooting

* **CUDA OOM:** If you encounter Out-Of-Memory errors on the GPU, try reducing the `BATCH_SIZE` in `main.py` (e.g., 64 or 128).
* **Matplotlib Errors:** If running on a headless server, the code automatically switches to the `Agg` backend to save plots without a display.
