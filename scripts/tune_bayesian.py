#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for BayesianRNN on TEP fault detection.

Uses Pyro SVI (Stochastic Variational Inference) with an AutoNormal guide
and TraceMeanField_ELBO loss. Predictions are made by averaging logits over
multiple posterior samples.

Usage:
    python scripts/tune_bayesian.py --config configs/config.yaml
"""

import sys
import json
import argparse
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoNormal

from configs.config_loader import load_config
from src.models.bayesian_rnn import BayesianRNN
from scripts.tune import seed_everything


# ======================================================================
# SVI training loop
# ======================================================================
def svi_train(svi, train_loader, max_epochs, device, patience=None, verbose=False):
    """
    Train with SVI for up to max_epochs, optionally with early stopping on ELBO.

    Returns
    -------
    best_epoch_elbo : float   — best (most negative) mean ELBO seen
    epoch_elbos     : list[float]
    """
    best_elbo = float('inf')
    patience_counter = 0
    epoch_elbos = []

    for epoch in range(max_epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = svi.step(X_batch, y_batch)
            total_loss += loss
            n_batches += 1

        avg_elbo = total_loss / max(n_batches, 1)
        epoch_elbos.append(avg_elbo)

        if verbose:
            print(f"    Epoch {epoch+1:3d}/{max_epochs} — ELBO: {avg_elbo:.2f}")

        if patience is not None:
            if avg_elbo < best_elbo:
                best_elbo = avg_elbo
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break

    return min(epoch_elbos), epoch_elbos


# ======================================================================
# Prediction via posterior sampling
# ======================================================================
def svi_predict(model, X_tensor, n_samples, device, batch_size):
    """
    Predict by averaging logits over n_samples posterior samples.

    Returns
    -------
    y_pred : ndarray (n_windows,)
    y_prob : ndarray (n_windows, output_dim)  — mean softmax probs
    """
    model.eval()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            # Sample n_samples times and average softmax probs
            batch_samples = []
            for _ in range(n_samples):
                logits = model(X_batch)   # samples weights from guide context
                probs = torch.softmax(logits, dim=1)
                batch_samples.append(probs)
            mean_probs = torch.stack(batch_samples).mean(0)  # (batch, output_dim)
            all_probs.append(mean_probs.cpu())

    y_prob = torch.cat(all_probs, dim=0).numpy()   # (n_windows, output_dim)
    y_pred = y_prob.argmax(axis=1)
    return y_pred, y_prob


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Optuna tuning for BayesianRNN (Pyro SVI)')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--n-samples-eval', type=int, default=10,
                        help='Posterior samples for val/test evaluation during tuning')
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.random_seed
    seed_everything(seed)

    print("=" * 70)
    print("  Bayesian RNN — Optuna Tuning (Pyro SVI)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load windowed data
    # ------------------------------------------------------------------
    ws = config.window_size
    stride = config.stride
    windows_dir = Path(config.output_dir) / 'windows'

    print(f"\nLoading data from {windows_dir} (w={ws}, s={stride})...")

    train = np.load(windows_dir / f'train_windows_w{ws}_s{stride}.npz',
                    allow_pickle=True)
    val = np.load(windows_dir / f'val_windows_w{ws}_s{stride}.npz',
                  allow_pickle=True)
    test = np.load(windows_dir / f'test_windows_w{ws}_s{stride}.npz',
                   allow_pickle=True)

    X_train = torch.tensor(train['X'], dtype=torch.float32)
    y_train = torch.tensor(train['y'], dtype=torch.long)
    X_val   = torch.tensor(val['X'],   dtype=torch.float32)
    y_val   = torch.tensor(val['y'],   dtype=torch.long)
    X_test  = torch.tensor(test['X'],  dtype=torch.float32)
    y_test  = torch.tensor(test['y'],  dtype=torch.long)

    input_dim  = X_train.shape[1]
    output_dim = int(torch.cat([y_train, y_val, y_test]).max().item()) + 1

    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | "
          f"Test: {X_test.shape[0]:,}")
    print(f"  Input dim: {input_dim} | Output classes: {output_dim}")

    batch_size  = config.batch_size
    max_epochs  = config.max_epochs
    patience    = config.early_stopping_patience
    n_trials    = config.n_trials
    ss          = config.tuning_search_space
    n_samples   = args.n_samples_eval

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # 2. Output directory
    # ------------------------------------------------------------------
    results_root = Path(config.results_dir)
    out_dir = results_root / 'bayesian_rnn'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Optuna objective
    # ------------------------------------------------------------------
    def objective(trial):
        # Clear Pyro param store to avoid leakage between trials
        pyro.clear_param_store()
        torch.manual_seed(seed + trial.number)

        print(f"\n  --- Trial {trial.number + 1}/{n_trials} ---")

        hidden_layers = trial.suggest_int(
            'hidden_layers', ss['hidden_layers']['min'], ss['hidden_layers']['max'])
        hidden_dim = trial.suggest_int(
            'hidden_dim', ss['hidden_dim']['min'], ss['hidden_dim']['max'])
        lr = trial.suggest_float(
            'lr', ss['learning_rate']['min'], ss['learning_rate']['max'],
            log=ss['learning_rate'].get('log', True))
        dropout_prob = trial.suggest_float(
            'dropout_prob', ss['dropout_prob']['min'], ss['dropout_prob']['max'])

        print(f"    Params: hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, "
              f"lr={lr:.2e}, dropout_prob={dropout_prob:.3f}")

        model = BayesianRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            seq_len=config.window_size,
        ).to(device)

        guide = AutoNormal(model)
        optimizer = pyro.optim.Adam({'lr': lr})
        svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

        _g = torch.Generator().manual_seed(seed + trial.number)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True, generator=_g)

        svi_train(svi, train_loader, max_epochs=max_epochs,
                  device=device, patience=patience)

        # Validation accuracy via posterior sampling
        y_pred, _ = svi_predict(model, X_val, n_samples=n_samples,
                                 device=device, batch_size=batch_size)
        val_acc = (y_pred == y_val.numpy()).mean()
        print(f"    Result: val_acc={val_acc:.4f}")
        return val_acc

    # ------------------------------------------------------------------
    # 4. Optuna study (SQLite persistence)
    # ------------------------------------------------------------------
    storage_path = out_dir / 'bayesian_rnn_study.db'
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f'sqlite:///{storage_path}',
        study_name='bayesian_rnn',
        load_if_exists=True,
    )

    n_completed = len(study.trials)
    n_remaining = max(0, n_trials - n_completed)

    if n_completed > 0:
        print(f"\n  Resuming: {n_completed} trials done, {n_remaining} remaining")

    print(f"\n{'='*70}")
    print(f"  Starting Optuna — {n_trials} trials")
    print(f"{'='*70}\n")

    start_time = time.time()

    def progress_callback(study, trial):
        n = trial.number + 1
        if n % 10 == 0 or n == n_trials:
            elapsed = time.time() - start_time
            print(f"  Trial {n:3d}/{n_trials} | "
                  f"Best val_acc: {study.best_value:.4f} | "
                  f"Elapsed: {elapsed:.0f}s")

    study.optimize(objective, n_trials=n_remaining,
                   callbacks=[progress_callback])

    elapsed_total = time.time() - start_time
    best_params = study.best_params
    best_val_acc = study.best_value

    print(f"\n{'='*70}")
    print(f"  Tuning complete in {elapsed_total:.1f}s")
    print(f"  Best val_accuracy: {best_val_acc:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 5. Retrain best model from scratch
    # ------------------------------------------------------------------
    print(f"\n  Retraining best model from scratch...")
    pyro.clear_param_store()
    seed_everything(seed)

    best_model = BayesianRNN(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        hidden_layers=best_params['hidden_layers'],
        output_dim=output_dim,
        dropout_prob=best_params['dropout_prob'],
        seq_len=config.window_size,
    ).to(device)

    guide = AutoNormal(best_model)
    optimizer = pyro.optim.Adam({'lr': best_params['lr']})
    svi = SVI(best_model, guide, optimizer, loss=TraceMeanField_ELBO())

    _g_retrain = torch.Generator().manual_seed(seed)
    train_loader_final = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, generator=_g_retrain)

    _, _ = svi_train(svi, train_loader_final, max_epochs=max_epochs,
                     device=device, patience=patience, verbose=True)

    # Validation accuracy after retraining
    y_pred_val, _ = svi_predict(best_model, X_val, n_samples=n_samples,
                                 device=device, batch_size=batch_size)
    best_val_acc_retrain = (y_pred_val == y_val.numpy()).mean()

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred_test, _ = svi_predict(best_model, X_test, n_samples=n_samples,
                                  device=device, batch_size=batch_size)
    test_acc = (y_pred_test == y_test.numpy()).mean()
    print(f"\n  Test accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    params_path = out_dir / 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: {params_path}")

    model_path = out_dir / 'best_model.pt'
    torch.save(best_model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    guide_path = out_dir / 'best_guide.pt'
    torch.save(pyro.get_param_store().get_state(), guide_path)
    print(f"  Saved: {guide_path}")

    preds_path = out_dir / 'predictions.npz'
    np.savez(preds_path, y_pred=y_pred_test, y_true=y_test.numpy())
    print(f"  Saved: {preds_path}")

    metrics = {
        'test_accuracy':  float(test_acc),
        'val_accuracy':   float(best_val_acc_retrain),
        'n_trials':       n_trials,
        'best_trial':     study.best_trial.number,
        'n_samples_eval': n_samples,
    }
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {out_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
