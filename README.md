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
| BayesianRNN | `bayesian_rnn` | Bayesian RNN with weight priors via Pyro SVI (see note below) |

The KAN variants and standard ANN baselines (MLP, CNN, RNN, LSTM) are registered in `src/models/__init__.py` and share the same training pipeline (`tune.py`, `train_best.py`). **BayesianRNN is handled separately** — see the dedicated sections below.

> **Why BayesianRNN has separate scripts:** Unlike the other models which are trained with standard cross-entropy loss and `torch.optim.Adam`, BayesianRNN places Normal(0,1) priors over all RNN and output layer weights and is trained using Pyro's Stochastic Variational Inference (SVI) with a `TraceMeanField_ELBO` objective and an `AutoNormal` variational guide. This requires a fundamentally different training loop (SVI steps, param store management, ELBO tracking) that is incompatible with the standard `tune.py` / `train_best.py` pipeline. Predictions are also made differently: instead of a single forward pass, logits are averaged over multiple posterior samples to produce stable softmax probabilities.

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
        -Drop features corresponding with analyzer measurements (Not realistic to have in real-life application of model)
    - Scaling (Z-scale)
        -Perform scaling on training dataset with transformation applied to validation and testing dataset
3) Generate sliding windows
    -Create sliding windows of length 5 with a stride of 1
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
# 3a. Tune all standard models (Optuna, 50 trials each)
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
# 3b. Tune BayesianRNN (separate script — Pyro SVI, 50 trials)
python scripts/tune_bayesian.py
```
Outputs to `results_N50_tr30_v10_te10/bayesian_rnn/`:
- `best_params.json` — best hyperparameters
- `best_model.pt` — model state dict retrained with best hyperparameters
- `best_guide.pt` — Pyro variational parameters (AutoNormal guide state)
- `predictions.npz` — test set predictions (`y_pred`, `y_true`) — note: `y_prob` not saved at this stage
- `metrics.json` — test accuracy, val accuracy, trial count, posterior sample count
- `bayesian_rnn_study.db` — SQLite Optuna study (resumable if interrupted)

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N50_tr30_v10_te10/<model>/`:
- `eval_metrics.json` — full metrics (see Evaluation Outputs below)
- `loss_curve.png` — training loss curve
- `confusion_matrix.png` — row-normalised confusion matrix heatmap
- `time_series_IDV{k}.png` — probability time-series plot for each fault class (27 plots)

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
# 3a. Train all standard models with tuned hyperparameters (100 epochs, no early stopping)
#     --params-dir points to best_params.json files from Experiment 1
python scripts/train_best.py --all --params-dir results_N50_tr30_v10_te10
```
This trains all 8 standard models (4 KAN variants + 4 ANN baselines) sequentially.
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
# 3b. Train BayesianRNN with tuned hyperparameters (separate script — Pyro SVI)
python scripts/train_best_bayesian.py --params-dir results_N50_tr30_v10_te10
# Use more posterior samples for stable final evaluation (default: 100)
python scripts/train_best_bayesian.py --params-dir results_N50_tr30_v10_te10 --n-samples 100
```
Outputs to `results_N200_tr160_v0_te40/bayesian_rnn/`:
- `best_model.pt` — final model state dict
- `best_guide.pt` — Pyro variational parameters required to reconstruct the posterior
- `predictions.npz` — test set predictions (`y_pred`, `y_true`, `y_prob`, `Run_ID`, `start_idx`, `end_idx`)
- `metrics.json` — test accuracy, epoch count, ELBO training curve, posterior sample count

```bash
# 4. Evaluate results
python scripts/evaluate.py
```
Outputs per model to `results_N200_tr160_v0_te40/<model>/`:
- `eval_metrics.json` — full metrics (see Evaluation Outputs below)
- `loss_curve.png` — training loss curve
- `confusion_matrix.png` — row-normalised confusion matrix heatmap
- `time_series_IDV{k}.png` — probability time-series plot for each fault class (27 plots)

---

## Evaluation Outputs

`eval_metrics.json` contains the following fields for each model:

| Field | Description |
|---|---|
| `accuracy` | Overall test accuracy |
| `macro_f1` | Macro-averaged F1 score |
| `per_class_f1` | F1 score for each class (dict keyed by class integer) |
| `confusion_matrix` | Raw confusion matrix (list of lists) |
| `classes` | Ordered list of class integers present in the test set |
| `mean_conf_correct` | Mean softmax confidence on correctly predicted windows |
| `mean_conf_wrong` | Mean softmax confidence on incorrectly predicted windows |
| `alarm_threshold` | Fault-probability threshold used for alarm metrics (0.90) |
| `false_alarm_rate` | FAR — proportion of healthy windows that trigger alarm |
| `correct_normal_rate` | CNR — proportion of healthy windows correctly not alarmed (= 1 − FAR) |
| `detection_rate` | FDR — proportion of fault windows that trigger alarm (all fault classes) |
| `miss_rate` | MR — proportion of fault windows that do not trigger alarm (= 1 − FDR) |
| `per_class_detection_rate` | FDR broken down by individual fault class IDV# (dict keyed by class integer) |
| `mean_top2_margin` | Mean gap between top-1 and top-2 softmax probabilities |
| `frac_ambiguous` | Fraction of windows where top-2 margin < 10 pp |

**Alarm metric definitions** (threshold: fault probability = 1 − P(class 0) > 0.90):

```
FAR + CNR = 1.0   (all healthy windows accounted for)
FDR + MR  = 1.0   (all fault windows accounted for)
```

**Time-series plots** (`time_series_IDV{k}.png`): for each fault class, shows the mean ± 1 std of P(healthy) and P(IDV#) across all 40 test runs. P(healthy) is expected to dominate before timestep 600 (fault insertion) and collapse thereafter as P(IDV#) rises.
