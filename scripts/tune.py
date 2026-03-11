#!/usr/bin/env python3
# Optuna hyperparameter tuning for KAN variants on TEP fault detection.
# Usage: python scripts/tune.py --model wavelet_kan --config configs/config.yaml

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

from configs.config_loader import load_config
from src.models import MODEL_REGISTRY


# ======================================================================
# Helper: variant-specific hyperparameters
# ======================================================================
def seed_everything(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


VARIANT_PARAMS = {
    'wavelet_kan':   ['wavelet_type'],
    'fourier_kan':   ['gridsize', 'smooth_initialization'],
    'fast_kan':      ['num_grids', 'grid_min', 'grid_max'],
    'efficient_kan': ['grid_size', 'spline_order'],
    'mlp':           ['dropout_prob', 'weight_decay'],
    'cnn':           ['num_filters', 'kernel_size', 'dropout_prob', 'weight_decay'],
    'rnn':           ['dropout_prob', 'weight_decay'],
    'lstm':          ['dropout_prob', 'weight_decay'],
}


def sample_variant_params(trial, model_name, search_space):
    """Sample variant-specific hyperparameters from the Optuna trial."""
    kwargs = {}
    param_names = VARIANT_PARAMS.get(model_name, [])

    for name in param_names:
        spec = search_space.get(name)
        if spec is None:
            continue

        if 'choices' in spec:
            kwargs[name] = trial.suggest_categorical(name, spec['choices'])
        elif 'min' in spec and 'max' in spec:
            # Determine int vs float from the values
            if isinstance(spec['min'], float) or isinstance(spec['max'], float):
                kwargs[name] = trial.suggest_float(name, spec['min'], spec['max'])
            else:
                kwargs[name] = trial.suggest_int(name, spec['min'], spec['max'])

    return kwargs


# ======================================================================
# Training loop with early stopping
# ======================================================================
def train_model(model, train_loader, val_loader, lr, max_epochs, patience,
                device, seed, weight_decay=0.0, verbose=False):
    """Train with early stopping; returns (best_val_acc, best_state_dict)."""
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total

        if verbose:
            print(f"    Epoch {epoch+1:3d}/{max_epochs} — val_acc: {val_acc:.4f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

    return best_val_acc, best_state


# ======================================================================
# Evaluate on a dataset
# ======================================================================
def evaluate(model, loader, device):
    """Return (accuracy, y_pred, y_true) arrays."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    acc = (y_pred == y_true).mean()
    return acc, y_pred, y_true


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Optuna KAN tuning')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='KAN variant to tune')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    model_name = args.model
    config = load_config(args.config)
    seed_everything(config.random_seed)

    print("=" * 70)
    print(f"  KAN Hyperparameter Tuning — {model_name}")
    print("=" * 70)

    # Load windowed data
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
    X_val = torch.tensor(val['X'], dtype=torch.float32)
    y_val = torch.tensor(val['y'], dtype=torch.long)
    X_test = torch.tensor(test['X'], dtype=torch.float32)
    y_test = torch.tensor(test['y'], dtype=torch.long)

    input_dim = X_train.shape[1]
    output_dim = int(torch.cat([y_train, y_val, y_test]).max().item()) + 1

    print(f"  Train: {X_train.shape[0]:,} windows | Val: {X_val.shape[0]:,} | "
          f"Test: {X_test.shape[0]:,}")
    print(f"  Input dim: {input_dim} | Output classes: {output_dim}")

    batch_size = config.batch_size
    seed = config.random_seed
    num_workers = config.num_workers
    persistent = num_workers > 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = device.type == 'cuda'
    print(f"  Device: {device}")
    # val/test loaders are deterministic (no shuffle) — create once here
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=persistent,
                            pin_memory=pin)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=persistent,
                             pin_memory=pin)

    # Search space from config
    ss = config.tuning_search_space
    max_epochs = config.max_epochs
    patience = config.early_stopping_patience

    # Optuna objective
    def objective(trial):
        print(f"\n  --- Trial {trial.number + 1}/{config.n_trials} ---")
        torch.manual_seed(seed)

        # Each trial gets its own deterministic but unique shuffle order.
        # Using seed + trial.number ensures reproducibility whether the study
        # runs fresh or is resumed from the SQLite file.
        _g = torch.Generator().manual_seed(seed + trial.number)
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=batch_size, shuffle=True,
                                  generator=_g, num_workers=num_workers,
                                  persistent_workers=persistent, pin_memory=pin)

        # Universal params
        hidden_layers = trial.suggest_int(
            'hidden_layers', ss['hidden_layers']['min'],
            ss['hidden_layers']['max'])
        hidden_dim = trial.suggest_int(
            'hidden_dim', ss['hidden_dim']['min'],
            ss['hidden_dim']['max'])
        lr = trial.suggest_float(
            'lr', ss['learning_rate']['min'],
            ss['learning_rate']['max'], log=ss['learning_rate'].get('log', True))

        # Variant-specific params
        variant_kwargs = sample_variant_params(trial, model_name, ss)
        wd = variant_kwargs.pop('weight_decay', 0.0)
        if model_name in ('rnn', 'lstm', 'cnn'):
            variant_kwargs['seq_len'] = config.window_size

        print(f"    Params: hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, "
              f"lr={lr:.2e}, wd={wd:.2e}, {variant_kwargs}")

        # Build model
        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            **variant_kwargs,
        )

        # Train with early stopping
        val_acc, _ = train_model(
            model, train_loader, val_loader,
            lr=lr, max_epochs=max_epochs, patience=patience,
            device=device, seed=seed, weight_decay=wd, verbose=False,
        )

        print(f"    Result: val_acc={val_acc:.4f}")

        return val_acc

    # Run study with SQLite persistence so it can be resumed
    results_root = Path(config.results_dir)
    out_dir = results_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_path = out_dir / f'{model_name}_study.db'

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f'sqlite:///{storage_path}',
        study_name=model_name,
        load_if_exists=True,
    )

    n_completed = len(study.trials)
    n_remaining = max(0, config.n_trials - n_completed)

    if n_completed > 0:
        print(f"\n  Resuming: {n_completed} trials already done, "
              f"{n_remaining} remaining")

    print(f"\n{'='*70}")
    print(f"  Starting Optuna optimization — {config.n_trials} trials")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Callback for progress logging
    def progress_callback(study, trial):
        n = trial.number + 1
        if n % 10 == 0 or n == config.n_trials:
            elapsed = time.time() - start_time
            print(f"  Trial {n:3d}/{config.n_trials} | "
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

    # Retrain best config from scratch on full train set
    print(f"\n  Retraining best model from scratch...")

    # Separate universal and variant-specific params
    variant_kwargs = {}
    for key in VARIANT_PARAMS.get(model_name, []):
        if key in best_params:
            variant_kwargs[key] = best_params[key]
    wd = variant_kwargs.pop('weight_decay', 0.0)
    if model_name in ('rnn', 'lstm', 'cnn'):
        variant_kwargs['seq_len'] = config.window_size

    ModelClass = MODEL_REGISTRY[model_name]
    best_model = ModelClass(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        hidden_layers=best_params['hidden_layers'],
        output_dim=output_dim,
        **variant_kwargs,
    )

    # Fresh loader for the retrain — always starts from seed regardless of
    # how many trials ran or whether the study was resumed mid-way.
    _g_retrain = torch.Generator().manual_seed(seed)
    train_loader_final = DataLoader(TensorDataset(X_train, y_train),
                                    batch_size=batch_size, shuffle=True,
                                    generator=_g_retrain, num_workers=num_workers,
                                    persistent_workers=persistent, pin_memory=pin)

    best_val_acc_retrain, best_state = train_model(
        best_model, train_loader_final, val_loader,
        lr=best_params['lr'],
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        seed=seed,
        weight_decay=wd,
        verbose=True,
    )

    # Load best state for evaluation
    best_model.load_state_dict(best_state)
    best_model = best_model.to(device)

    # Evaluate on held-out test set
    test_acc, y_pred, y_true = evaluate(best_model, test_loader, device)
    print(f"\n  Test accuracy: {test_acc:.4f}")

    # Save outputs
    # best_params.json
    params_path = out_dir / 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: {params_path} ({params_path.stat().st_size:,} bytes)")

    # best_model.pt
    model_path = out_dir / 'best_model.pt'
    torch.save(best_state, model_path)
    print(f"  Saved: {model_path} ({model_path.stat().st_size:,} bytes)")

    # predictions.npz
    preds_path = out_dir / 'predictions.npz'
    np.savez(preds_path, y_pred=y_pred, y_true=y_true)
    print(f"  Saved: {preds_path} ({preds_path.stat().st_size:,} bytes)")

    # metrics.json
    metrics = {
        'test_accuracy': float(test_acc),
        'val_accuracy': float(best_val_acc_retrain),
        'n_trials': config.n_trials,
        'best_trial': study.best_trial.number,
    }
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path} ({metrics_path.stat().st_size:,} bytes)")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {out_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()