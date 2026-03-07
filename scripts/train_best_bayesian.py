#!/usr/bin/env python3
"""
Train BayesianRNN with tuned hyperparameters on the full dataset (Experiment 2).

Loads best_params.json from Experiment 1, trains for a fixed number of epochs
with Pyro SVI (no early stopping, no validation split), and evaluates on the
test set using n_samples posterior samples for stable predictions.

Usage:
    python scripts/train_best_bayesian.py --config configs/config.yaml
    python scripts/train_best_bayesian.py --config configs/config.yaml --n-samples 100
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import pyro
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoNormal

from configs.config_loader import load_config
from src.models.bayesian_rnn import BayesianRNN
from scripts.tune import seed_everything


# ======================================================================
# Fixed-epoch SVI training (no early stopping)
# ======================================================================
def svi_train_fixed(svi, train_loader, max_epochs, device, verbose=True):
    """
    Train with SVI for exactly max_epochs epochs.

    Returns
    -------
    epoch_elbos : list[float] — mean ELBO loss per epoch
    """
    epoch_elbos = []
    epoch_bar = tqdm(range(max_epochs), desc="    Training", unit="epoch",
                     disable=not verbose, dynamic_ncols=True)
    for epoch in epoch_bar:
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
        epoch_bar.set_postfix(elbo=f"{avg_elbo:.2f}")

    return epoch_elbos


# ======================================================================
# Prediction via posterior sampling (returns probs for alarm analysis)
# ======================================================================
def svi_predict(model, X_tensor, n_samples, device, batch_size, verbose=True):
    """
    Predict by averaging softmax probabilities over n_samples posterior samples.

    Returns
    -------
    y_pred : ndarray (n_windows,)
    y_prob : ndarray (n_windows, output_dim) — mean softmax probs
    """
    model.eval()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        bar = tqdm(loader, desc="    Evaluating", unit="batch",
                   leave=False, disable=not verbose)
        for (X_batch,) in bar:
            X_batch = X_batch.to(device)
            batch_samples = []
            for _ in range(n_samples):
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                batch_samples.append(probs)
            mean_probs = torch.stack(batch_samples).mean(0)
            all_probs.append(mean_probs.cpu())

    y_prob = torch.cat(all_probs, dim=0).numpy()
    y_pred = y_prob.argmax(axis=1)
    return y_pred, y_prob


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train BayesianRNN with tuned hyperparameters (Experiment 2)')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: config.results_dir)')
    parser.add_argument('--params-dir', type=str, default=None,
                        help='Directory with Exp 1 best_params.json '
                             '(default: config.results_dir)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Posterior samples for test evaluation (default: 100)')
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.random_seed
    seed_everything(seed)

    output_dir = Path(args.output_dir or config.results_dir)
    params_dir = Path(args.params_dir or config.results_dir)
    windows_dir = Path(config.output_dir) / 'windows'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  BayesianRNN — Training with Tuned Hyperparameters")
    print("=" * 70)
    print(f"  Params from: {params_dir / 'bayesian_rnn' / 'best_params.json'}")
    print(f"  Output to:   {output_dir / 'bayesian_rnn'}/")
    print(f"  Windows:     {windows_dir}")
    print(f"  Device:      {device}")
    print(f"  Max epochs:  {config.max_epochs}")
    print(f"  N samples:   {args.n_samples}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load best hyperparameters from Experiment 1
    # ------------------------------------------------------------------
    params_path = params_dir / 'bayesian_rnn' / 'best_params.json'
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found.\n"
            "Run scripts/tune_bayesian.py (Experiment 1) first."
        )
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"\n  Loaded params: {json.dumps(best_params)}")

    # ------------------------------------------------------------------
    # 2. Load train + test windows (no validation for Exp 2)
    # ------------------------------------------------------------------
    ws     = config.window_size
    stride = config.stride
    train_npz = windows_dir / f'train_windows_w{ws}_s{stride}.npz'
    test_npz  = windows_dir / f'test_windows_w{ws}_s{stride}.npz'

    for p in [train_npz, test_npz]:
        if not p.exists():
            raise FileNotFoundError(
                f"Window file not found: {p}\n"
                "Run scripts/create_windows.py first."
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

    # ------------------------------------------------------------------
    # 3. Build DataLoaders
    # ------------------------------------------------------------------
    batch_size = config.batch_size
    _g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, generator=_g)

    # ------------------------------------------------------------------
    # 4. Build model and guide
    # ------------------------------------------------------------------
    pyro.clear_param_store()

    model = BayesianRNN(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        hidden_layers=best_params['hidden_layers'],
        output_dim=output_dim,
        dropout_prob=best_params['dropout_prob'],
        seq_len=config.window_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    guide     = AutoNormal(model)
    optimizer = pyro.optim.Adam({'lr': best_params['lr']})
    svi       = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

    # ------------------------------------------------------------------
    # 5. Train for fixed epochs
    # ------------------------------------------------------------------
    max_epochs = config.max_epochs
    print(f"\n  Training for {max_epochs} epochs (no early stopping)...")
    t0 = time.time()

    epoch_elbos = svi_train_fixed(svi, train_loader, max_epochs=max_epochs,
                                   device=device, verbose=True)

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    print(f"\n  Evaluating with {args.n_samples} posterior samples...")
    y_pred, y_prob = svi_predict(model, X_test, n_samples=args.n_samples,
                                  device=device, batch_size=batch_size)
    test_acc = (y_pred == y_test.numpy()).mean()
    print(f"  Test accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    out_dir = output_dir / 'bayesian_rnn'
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / 'best_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    guide_path = out_dir / 'best_guide.pt'
    torch.save(pyro.get_param_store().get_state(), guide_path)
    print(f"  Saved: {guide_path}")

    preds_path = out_dir / 'predictions.npz'
    save_kwargs = dict(y_pred=y_pred, y_true=y_test.numpy(), y_prob=y_prob)
    for key in ('Run_ID', 'start_idx', 'end_idx'):
        if key in test_data:
            save_kwargs[key] = test_data[key]
    np.savez(preds_path, **save_kwargs)
    print(f"  Saved: {preds_path}")

    metrics = {
        'test_accuracy':     float(test_acc),
        'max_epochs':        max_epochs,
        'best_params':       best_params,
        'train_windows':     int(X_train.shape[0]),
        'test_windows':      int(X_test.shape[0]),
        'train_elbo_curve':  [float(v) for v in epoch_elbos],
        'n_samples':         args.n_samples,
    }
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
