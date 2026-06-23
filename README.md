Mashroor Rahman's Undergraduate Honours Thesis Project (ChE 4Y04) on the evaluation of KAN variants for fault detection and diagnosis of the Tenessee Eastman Process. This work is conducted under the supervision of Dr. Giancarlo Dalle Ave, with guidance from PhD candidate Jose Daniel Rojas Dorantes, in the Department of Chemical Engineering at McMaster University.

## Project Structure
scripts/      → runnable pipeline entry points
src/          → core ML modules and logic
tests/        → validation and unit tests
requirements.txt → Python dependencies

## Models

Eight models are implemented and evaluated as part of this project:

**KAN variants** (primary subject of study):
| Model | Key | Description |
|---|---|---|
| EfficientKAN | `efficient_kan` | B-spline-based KAN with learnable grid and spline order |
| FourierKAN | `fourier_kan` | KAN using Fourier basis functions |
| WaveletKAN | `wavelet_kan` | KAN using wavelet basis functions |
| FastKAN | `fast_kan` | KAN using radial basis functions on a learnable grid |

**ANN baselines** (for comparison):
| Model | Key | Description |
|---|---|---|
| MLP | `mlp` | Multi-layer perceptron with dropout |
| CNN | `cnn` | 1-D convolutional network with dropout |
| RNN | `rnn` | Recurrent neural network with dropout |
| LSTM | `lstm` | Long short-term memory network with dropout |

All models are registered in `src/models/__init__.py` and share the same training pipeline (`tune.py`, `train_best.py`).

## Setup
1) Install dependencies: python -m pip install -r requirements.txt
2) Place the [raw H5 file](https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936) (Mode 1 only) at the path set by `data.raw_source` in `configs/config.yaml` (default: `data\raw\TEP_Mode1.h5`)
3) Edit `configs/config.yaml` (random seed, splits, windowing, tuning search space)


## Pipeline Overview
1) Load and label data from H5 (`src/preprocessing/tep_data_processor.py`)
    - Randomly selects `total_runs` runs per IDV and splits them into train/val/test per the `splits` block in `configs/config.yaml` (e.g. 30/10/10 for Experiment 1, 160/0/40 for Experiment 2)
    - Label data with header row from "Processdata_Labels" and "Additional_Meas_Labels"
    - Add target column for classifcation of timestep: healthy (0), IDV(X) (X)
        -Healthy data is indices 0-599
        -Faulty data is indices 600-2001
2) Perform pre-processing and feature engineering
    - Feature engineering
        -Drop features with zero variance
        -Drop features corresponding with analyzer measurements (Not realistic to have in real-life application of model)
    - Scaling (Z-scale)
        -Perform scaling on training dataset with transformation applied to validation and testing dataset
3) Generate sliding windows
    -Create sliding windows per the `windowing` block in `configs/config.yaml` (default: length 5, stride 1)
4) Train model
5) Hyperparameter tuning
6) Evaluate performance

---

## Experiment 1 — Hyperparameter Tuning

Uses 50 runs per IDV (30 train / 10 val / 10 test) to find the best hyperparameters
for each model via Optuna (50 trials per model).

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
# 3. Tune all models (Optuna, 50 trials each)
# KAN variants
python scripts/tune.py --model efficient_kan
python scripts/tune.py --model fourier_kan
python scripts/tune.py --model wavelet_kan
python scripts/tune.py --model fast_kan
# ANN baselines
python scripts/tune.py --model mlp
python scripts/tune.py --model cnn
python scripts/tune.py --model rnn
python scripts/tune.py --model lstm
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `best_params.json` — best hyperparameters found by Optuna
- `best_model.pt` — model retrained with best hyperparameters
- `predictions.npz` — test set predictions (`y_pred`, `y_true`, `y_prob`, `Run_ID`, `start_idx`, `end_idx`)
- `metrics.json` — val accuracy, training loss curve, trial count, best trial number

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `eval_metrics.json` — full metrics (see Evaluation Outputs below)
- `loss_curve.png` — training loss curve
- `confusion_matrix.png` — row-normalised confusion matrix heatmap

Output to `results_N50_tr30_v10_te10/`:
- `model_evaluation.xlsx` — cross-model comparison workbook (see Evaluation Outputs below)

```bash
# 5. (Optional) Generate per-fault probability time-series plots
python scripts/plot_time_series.py
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `time_series_IDV{k}.png` — probability time-series plot for each fault class

---

## Experiment 2 — Full Dataset Training (200 runs/IDV)

Trains all eight models on the full dataset (200 runs per IDV, 160 train / 40 test)
using the best hyperparameters found in Experiment 1. No hyperparameter search is
performed — `best_params.json` from `results/` is loaded directly.

**Prerequisite:** Experiment 1 must be completed. The following files must exist for each model:
```
results_N50_tr30_v10_te10/<model>/best_params.json
```
where `<model>` is each of: `efficient_kan`, `fourier_kan`, `wavelet_kan`, `fast_kan`, `mlp`, `cnn`, `rnn`, `lstm`.

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
# 3. Train all models with tuned hyperparameters (early stopping on train loss)
#    --params-dir points to best_params.json files from Experiment 1
python scripts/train_best.py --all --params-dir results_N50_tr30_v10_te10
```
This trains all 8 models (4 KAN variants + 4 ANN baselines) sequentially.
To train a single model instead:
```bash
python scripts/train_best.py --model wavelet_kan --params-dir results_N50_tr30_v10_te10
python scripts/train_best.py --model lstm        --params-dir results_N50_tr30_v10_te10
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `best_model.pt` — final trained model weights
- `predictions.npz` — test set predictions (`y_pred`, `y_true`, `y_prob`, `Run_ID`, `start_idx`, `end_idx`)
- `metrics.json` — test accuracy, epoch count, training loss curve

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `eval_metrics.json` — full metrics (see Evaluation Outputs below)
- `loss_curve.png` — training loss curve
- `confusion_matrix.png` — row-normalised confusion matrix heatmap

Output to `results_N200_tr160_v0_te40/`:
- `model_evaluation.xlsx` — cross-model comparison workbook (see Evaluation Outputs below)

```bash
# 5. (Optional) Generate per-fault probability time-series plots
python scripts/plot_time_series.py
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `time_series_IDV{k}.png` — probability time-series plot for each fault class

---

## Evaluation Outputs

`eval_metrics.json` contains the following fields for each model:

| Field | Description |
|---|---|
| `accuracy` | Overall test accuracy |
| `macro_f1` | Macro-averaged F1 score |
| `per_class_f1` | F1 score for each class |
| `confusion_matrix` | Raw confusion matrix |
| `classes` | Ordered list of class integers present in the test set |
| `mean_conf_correct` | Mean softmax confidence on correctly predicted windows |
| `mean_conf_wrong` | Mean softmax confidence on incorrectly predicted windows |
| `alarm_threshold` | Fault-probability threshold used for alarm metrics (0.90) |
| `false_alarm_rate` | FAR — proportion of healthy windows that trigger alarm |
| `correct_normal_rate` | CNR — proportion of healthy windows correctly not alarmed (= 1 − FAR) |
| `detection_rate` | FDR — proportion of fault windows that trigger alarm (all fault classes) |
| `miss_rate` | MR — proportion of fault windows that do not trigger alarm (= 1 − FDR) |
| `per_class_detection_rate` | FDR broken down by individual fault class IDV# |
| `mean_top2_margin` | Mean gap between top-1 and top-2 softmax probabilities |
| `frac_ambiguous` | Fraction of windows where top-2 margin < 10 pp |
| `timing_metrics` | Per-fault-class FDet/FDiag timing (see below) |
| `best_params` | Hyperparameters used for this run (from `best_params.json`, if present) |
| `val_accuracy`, `n_trials`, `best_trial` | Optuna tuning metadata (Experiment 1 only; `null` for Experiment 2) |
| `epochs_trained`, `training_time_s` | Epochs run before early stopping and wall-clock training time |

**Alarm metric definitions** (threshold: fault probability = 1 − P(class 0) > 0.90 by default — see Sensitivity Analysis below for varying this):

```
FAR + CNR = 1.0   (all healthy windows accounted for)
FDR + MR  = 1.0   (all fault windows accounted for)
```

`timing_metrics` (per fault class IDV#, relative to fault insertion at index 600):
- `fdet_mean` / `fdet_std` — Fault Detection Time: first window where P(NOC) drops below the alarm threshold
- `fdiag_mean` / `fdiag_std` — Fault Diagnosis Time: first window where the probability of the *true* fault class exceeds 90%
- `fdiag_accuracy` — of all windows confidently diagnosed (any non-NOC class > 90%), the fraction diagnosed as the correct class

`model_evaluation.xlsx` (written by `evaluate.py`, one row/column per model) contains:
- **Model Comparison** — accuracy, macro F1, val accuracy, confidence, epochs, training time, best params
- **Per-Class F1** — F1 score per fault class, per model
- **Alarm Analysis** — FAR, CNR, FDR, MR, top-2 margin, per model
- **Per-Class Detection Rate** — detection rate per fault class, per model
- **Fault Detection Time** — FDet mean ± std and detected/total count per fault class, per model
- **Fault Diagnosis Time** — FDiag mean ± std, diagnosis accuracy, per fault class, per model

---

## Sensitivity Analysis

Sensitivity sweeps re-examine an **already-trained** model under different post-hoc or pipeline settings, isolating one variable at a time. Each sweep is documented in its own subsection below as it's implemented.

### Alarm-Threshold Sweep

Varies the alarm-onset cutoff — the model and its weights are untouched; this only changes how the *already-computed* softmax probabilities (`predictions.npz` → `y_prob`) are thresholded to decide "alarm vs. no alarm." No retraining or data regeneration is involved, since the threshold never enters the training objective.

By default, an alarm fires when `P(NOC) < 10%` (`alarm_threshold = 0.90` in `evaluate.py`). This sweep recomputes alarm/timing metrics at `P(NOC) < 5%, 7.5%, 10%, 12.5%, 15%` against the same saved predictions.

**Prerequisite:** `predictions.npz` must already exist for the target model (i.e. `train_best.py` or `tune.py` has been run for it).

```bash
# Sweep the default thresholds (5%, 7.5%, 10%, 12.5%, 15%) for WavKAN
python scripts/sensitivity_threshold.py --model wavelet_kan

# Or specify custom thresholds
python scripts/sensitivity_threshold.py --model wavelet_kan --thresholds 0.05,0.10,0.20
```

Output to `<results_dir>/<model>/threshold_sensitivity.xlsx` (e.g. `results_N200_tr160_v0_te40/wavelet_kan/threshold_sensitivity.xlsx`), one column per threshold instead of per model:
- **Alarm Analysis** — FAR, CNR, FDR, MR per threshold
- **Per-Class Detection Rate** — detection rate per fault class, per threshold
- **Fault Detection Time** — FDet mean ± std and detected/total count per fault class, per threshold
