#!/usr/bin/env python3
# Train KAN variants using best_params.json from tuning (Exp 1) on the full dataset.
# No Optuna, no val split — early stops on train loss.
# Usage: python scripts/train_best.py --model wavelet_kan
#        python scripts/train_best.py --all [--output-dir results_full --params-dir results]

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from configs.config_loader import load_config
from src.models import MODEL_REGISTRY
from scripts.tune import seed_everything, VARIANT_PARAMS


# ======================================================================
# Training loop — early stopping when loss stops improving
# ======================================================================
def train_early_stopping(model, train_loader, lr, max_epochs, patience, device, seed,
                         weight_decay=0.0, min_delta=1e-4, verbose=True):
    """Early-stop on train loss; returns (best_state_dict, epoch_losses)."""
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    epoch_losses = []

    epoch_bar = tqdm(range(max_epochs), desc="    Training", unit="epoch",
                     disable=not verbose, dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", best=f"{best_loss:.4f}",
                              patience=f"{epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            if verbose:
                tqdm.write(f"    Early stopping at epoch {epoch + 1} "
                           f"(no improvement for {patience} epochs)")
            break

    return best_state, epoch_losses


# ======================================================================
# Evaluation helper (reuses tune.py logic)
# ======================================================================
def evaluate(model, loader, device):
    """Return (accuracy, y_pred, y_true, y_prob) where y_prob is softmax over classes."""
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="    Evaluating", unit="batch", leave=False):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    y_prob = np.vstack(all_probs)          # (n_windows, n_classes)
    acc = (y_pred == y_true).mean()
    return acc, y_pred, y_true, y_prob


# ======================================================================
# Train one model variant
# ======================================================================
def train_one_variant(model_name, config, windows_dir, params_dir, output_dir, device):
    """Train one variant with its tuned params and save results to output_dir/model_name/."""
    seed = config.random_seed
    ws   = config.window_size
    stride = config.stride

    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    # Load best hyperparameters from tuning
    params_path = Path(params_dir) / model_name / 'best_params.json'
    if not params_path.exists():
        print(f"  ERROR: {params_path} not found — skipping {model_name}")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"  Loaded params: {json.dumps(best_params)}")

    # Load train + test windows
    train_npz = windows_dir / f'train_windows_w{ws}_s{stride}.npz'
    test_npz  = windows_dir / f'test_windows_w{ws}_s{stride}.npz'

    for p in [train_npz, test_npz]:
        if not p.exists():
            raise FileNotFoundError(
                f"Window file not found: {p}\n"
                f"Have you run scripts/create_windows.py first?"
            )

    train_data = np.load(train_npz, allow_pickle=True)
    test_data  = np.load(test_npz,  allow_pickle=True)

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.long)
    X_test  = torch.tensor(test_data['X'],  dtype=torch.float32)
    y_test  = torch.tensor(test_data['y'],  dtype=torch.long)

    input_dim  = X_train.shape[1]
    output_dim = int(torch.cat([y_train, y_test]).max().item()) + 1

    print(f"  Train: {X_train.shape[0]:,} windows | Test: {X_test.shape[0]:,}")
    print(f"  Input dim: {input_dim} | Output classes: {output_dim}")

    # DataLoaders
    batch_size   = config.batch_size
    num_workers  = config.num_workers
    persistent   = num_workers > 0
    pin          = device.type == 'cuda'
    _g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, generator=_g,
                              num_workers=num_workers, persistent_workers=persistent,
                              pin_memory=pin)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=persistent,
                              pin_memory=pin)

    # Build model from best params
    seed_everything(seed)

    variant_kwargs = {k: best_params[k]
                     for k in VARIANT_PARAMS.get(model_name, [])
                     if k in best_params}
    wd = variant_kwargs.pop('weight_decay', 0.0)
    if model_name in ('rnn', 'lstm', 'cnn'):
        variant_kwargs['seq_len'] = config.window_size

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        hidden_layers=best_params['hidden_layers'],
        output_dim=output_dim,
        **variant_kwargs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Train
    max_epochs = config.max_epochs
    patience   = config.early_stopping_patience
    print(f"\n  Training (max {max_epochs} epochs, early-stop patience={patience})...")
    t0 = time.time()

    final_state, epoch_losses = train_early_stopping(
        model, train_loader,
        lr=best_params['lr'],
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        seed=seed,
        weight_decay=wd,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # Evaluate on test set
    model.load_state_dict(final_state)
    model = model.to(device)
    test_acc, y_pred, y_true, y_prob = evaluate(model, test_loader, device)
    print(f"  Test accuracy: {test_acc:.4f}")

    # Save outputs
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / 'best_model.pt'
    torch.save(final_state, model_path)
    print(f"  Saved: {model_path}")

    preds_path = out_dir / 'predictions.npz'
    # Save predictions, probabilities, and window metadata so every window
    # can be mapped back to its (Run_ID, timestep) for alarm / latency analysis.
    # y_prob shape: (n_windows, n_classes)
    #   fault_prob[i]  = 1 - y_prob[i, 0]   (probability window is NOT healthy)
    #   confidence[i]  = y_prob[i, y_pred[i]] (confidence in the predicted class)
    save_kwargs = dict(y_pred=y_pred, y_true=y_true, y_prob=y_prob)
    for key in ('Run_ID', 'start_idx', 'end_idx'):
        if key in test_data:
            save_kwargs[key] = test_data[key]
    np.savez(preds_path, **save_kwargs)
    print(f"  Saved: {preds_path}")

    metrics = {
        'test_accuracy': float(test_acc),
        'epochs_trained': len(epoch_losses),
        'max_epochs': max_epochs,
        'early_stopping_patience': patience,
        'training_time_s': float(elapsed),
        'best_params': best_params,
        'train_windows': int(X_train.shape[0]),
        'test_windows': int(X_test.shape[0]),
        'train_loss_curve': [float(l) for l in epoch_losses],
    }
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    return metrics


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train KAN variants with tuned hyperparameters (no Optuna)'
    )
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Single variant to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all 4 variants sequentially')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: config.results_dir)')
    parser.add_argument('--params-dir', type=str, default=None,
                        help='Directory containing best_params.json files from Exp 1 '
                             '(default: config.results_dir)')
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error('Specify --model <name> or --all')

    config = load_config(args.config)

    output_dir = args.output_dir or config.results_dir
    params_dir = args.params_dir or config.results_dir
    windows_dir = Path(config.output_dir) / 'windows'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  KAN Training with Tuned Hyperparameters")
    print("=" * 70)
    print(f"  Config:      {args.config}")
    print(f"  Params from: {params_dir}")
    print(f"  Output to:   {output_dir}")
    print(f"  Windows:     {windows_dir}")
    print(f"  Device:      {device}")
    print(f"  Max epochs:  {config.max_epochs} (early-stop patience={config.early_stopping_patience})")
    print("=" * 70)

    variants = list(MODEL_REGISTRY.keys()) if args.all else [args.model]

    all_metrics = {}
    for variant in variants:
        metrics = train_one_variant(
            model_name=variant,
            config=config,
            windows_dir=windows_dir,
            params_dir=params_dir,
            output_dir=output_dir,
            device=device,
        )
        if metrics is not None:
            all_metrics[variant] = metrics

    # Final summary table
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<18s} {'Test Acc':>10s}")
    print(f"  {'-'*18} {'-'*10}")
    for variant, m in all_metrics.items():
        print(f"  {variant:<18s} {m['test_accuracy']:>10.4f}")
    print(f"{'='*70}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
