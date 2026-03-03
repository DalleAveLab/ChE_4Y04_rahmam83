Mashroor Rahman's Undergraduate Honours Thesis Project (ChE 4Y04) on the evaluation of KAN variants for fault detection and diagnosis of the Tenessee Eastman Process. This work is conducted under the supervision of Dr. Giancarlo Dalle Ave, with guidance from PhD candidate Jose Daniel Rojas Dorantes, in the Department of Chemical Engineering at McMaster University.

## Project Structure
scripts/      → runnable pipeline entry points  
src/          → core ML modules and logic  
tests/        → validation and unit tests  
requirements.txt → Python dependencies  

## Setup
1) Install dependencies: python -m pip install -r requirements.txt
2) Place the [raw H5 file](https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936) (Mode 1 only) in: `data/raw/tep_data.h5`
2) Edit config.yaml


## Pipeline Overview
1) Load and label data from H5 (load_data.py)
    - Loads "processdata" and "additional_meas" into a split of 30, 10, 10 runs for training, validation, and testing set.
    - Label data with header row from "Processdata_Labels" and "Additional_Meas_Labels"
    - Add target column for classifcation of timestep: healthy (0), IDV(X) (X)
        -Healthy data is indices 0-599
        -Faulty data is indices 600-2001
2) Perform pre-processing and feature engineering
    - Feature engineering
        -Drop features with zero variance
        -Drop features corresponding with analyzer measurements (Not realistic to have in time for real-life application of model)
    - Scaling (Z-scale)
        -Perform sclaing on training dataset with transformation applied to validation and testing dataset
3) Generate sliding windows
    -Create sliding windows of length 5 with a stride of 1
4) Train model
5) Hyperparameter tuning
6) Evaluate performance

---

## Experiment 1 — Hyperparameter Tuning

Uses 50 runs per IDV (30 train / 10 val / 10 test) to find the best hyperparameters
for each KAN variant via Optuna (30 trials per model).

**Config settings** (`configs/config.yaml`):
```yaml
data:
  processed_base_dir: 'data\processed'   # suffix auto-appended → data\processed_N50_tr30_v10_te10
splits:
  total_runs: 50
  train_runs: 30
  val_runs: 10
  test_runs: 10
models:
  results_base_dir: 'results'            # suffix auto-appended → results_N50_tr30_v10_te10
```

The output and results directories are **automatically derived** from the split settings — no manual path changes needed between experiments.

**Run order:**

```bash
# 1. Extract and preprocess data (50 runs/IDV, 30/10/10 split)
python scripts/run_pipeline.py
```
Outputs to `data/processed_N50_tr30_v10_te10/`:
- `feature_engineer.pkl` — fitted feature engineering pipeline
- `scaler.pkl` — fitted StandardScaler (fit on train only)
- `train_final.pkl` — scaled training DataFrame
- `val_final.pkl` — scaled validation DataFrame
- `test_final.pkl` — scaled test DataFrame
- `train_final_sample.csv` — first 10,000 rows of train for inspection

```bash
# 2. Create sliding windows (train, val, test)
python scripts/create_windows.py
```
Outputs to `data/processed_N50_tr30_v10_te10/windows/`:
- `train_windows_w5_s1.npz` — training windows (X, y, metadata)
- `val_windows_w5_s1.npz` — validation windows
- `test_windows_w5_s1.npz` — test windows

```bash
# 3. Tune each KAN variant (Optuna, 50 trials each)
python scripts/tune.py --model efficient_kan
python scripts/tune.py --model fourier_kan
python scripts/tune.py --model wavelet_kan
python scripts/tune.py --model fast_kan
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `best_params.json` — best hyperparameters found by Optuna
- `best_model.pt` — model retrained with best hyperparameters
- `predictions.npz` — test set predictions (`y_pred`, `y_true`, `y_prob`)
- `metrics.json` — val accuracy, training loss curve, trial count, best trial number

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `eval_metrics.json` — full metrics (accuracy, macro F1, per-class F1, confusion matrix, alarm stats)
- `loss_curve.png` — training loss curve plot

---

## Experiment 2 — Full Dataset Training (200 runs/IDV)

Trains all four KAN variants on the full dataset (200 runs per IDV, 160 train / 40 test)
using the best hyperparameters found in Experiment 1. No hyperparameter search is
performed — `best_params.json` from `results/` is loaded directly.

**Prerequisite:** Experiment 1 must be completed. The following files must exist:
```
results_N50_tr30_v10_te10/efficient_kan/best_params.json
results_N50_tr30_v10_te10/fourier_kan/best_params.json
results_N50_tr30_v10_te10/wavelet_kan/best_params.json
results_N50_tr30_v10_te10/fast_kan/best_params.json
```

**Config settings** (`configs/config.yaml`):
```yaml
data:
  processed_base_dir: 'data\processed'   # suffix auto-appended → data\processed_N200_tr160_v0_te40
splits:
  total_runs: 200
  train_runs: 160
  val_runs: 0
  test_runs: 40
models:
  results_base_dir: 'results'            # suffix auto-appended → results_N200_tr160_v0_te40
```

The output and results directories are **automatically derived** from the split settings — only the `splits` block needs to change between experiments.

**Run order:**

```bash
# 1. Extract and preprocess data (200 runs/IDV, 160/40 split, no validation)
python scripts/run_pipeline.py
```
Outputs to `data/processed_N200_tr160_v0_te40/`:
- `feature_engineer.pkl` — fitted feature engineering pipeline
- `scaler.pkl` — fitted StandardScaler (fit on train only)
- `train_final.pkl` — scaled training DataFrame (160 runs/IDV)
- `test_final.pkl` — scaled test DataFrame (40 runs/IDV)
- `train_final_sample.csv` — first 10,000 rows of train for inspection

```bash
# 2. Create sliding windows (train and test only — no val)
python scripts/create_windows.py
```
Outputs to `data/processed_N200_tr160_v0_te40/windows/`:
- `train_windows_w5_s1.npz` — training windows (X, y, metadata)
- `test_windows_w5_s1.npz` — test windows

```bash
# 3. Train all four models with tuned hyperparameters (100 epochs, no early stopping)
#    --params-dir points to best_params.json files from Experiment 1
python scripts/train_best.py --all --params-dir results_N50_tr30_v10_te10
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `best_model.pt` — final trained model weights
- `predictions.npz` — test set predictions (`y_pred`, `y_true`, `y_prob`)
- `metrics.json` — test accuracy, epoch count, training loss curve

To train a single variant instead of all four:
```bash
python scripts/train_best.py --model wavelet_kan --params-dir results_N50_tr30_v10_te10
```

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `eval_metrics.json` — full metrics (accuracy, macro F1, per-class F1, confusion matrix, alarm stats)
- `loss_curve.png` — training loss curve plot
