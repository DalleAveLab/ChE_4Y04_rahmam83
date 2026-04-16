#!/usr/bin/env python3
"""
Plot IDV14 class probability time series for a given model.

For each IDV14 test run, shows:
  - P(NOC)       — class 0, always included
  - P(top-1)     — highest mean probability across the run (excluding NOC)
  - P(top-2)     — second highest
  - P(top-3)     — third highest

A vertical dashed line marks the fault insertion point (timestep 600).

Usage:
    python tests/plot_idv14_top3_probs.py
    python tests/plot_idv14_top3_probs.py --model fourier_kan
    python tests/plot_idv14_top3_probs.py --model wavelet_kan --results-dir results_N50_tr30_v10_te10
    python tests/plot_idv14_top3_probs.py --model wavelet_kan --run IDV14_Run104
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FAULT_START = 600
NOC_CLASS   = 0


def pick_top3_classes(y_prob_run: np.ndarray) -> list[int]:
    """
    Return the 3 non-NOC class indices with the highest mean probability
    across the run, in descending order.
    """
    n_classes = y_prob_run.shape[1]
    mean_probs = y_prob_run.mean(axis=0)          # (n_classes,)
    mean_probs[NOC_CLASS] = -1.0                  # exclude NOC
    top3 = np.argsort(mean_probs)[::-1][:3].tolist()
    return top3


def plot_run(run_id: str, end_idx: np.ndarray, y_prob: np.ndarray,
             y_pred: np.ndarray, y_true: np.ndarray,
             out_path: Path) -> None:
    """Plot NOC + top-3 non-NOC probabilities for a single run."""

    order    = np.argsort(end_idx)
    x        = end_idx[order]
    probs    = y_prob[order]          # (n_windows, n_classes)
    top3     = pick_top3_classes(probs)

    lines = [(NOC_CLASS, 'P(NOC)',          '#2196F3', 2.0, '-')]
    colors = ['#F44336', '#FF9800', '#4CAF50']
    for rank, cls in enumerate(top3):
        lines.append((cls, f'P(IDV{cls})', colors[rank], 1.5, '-'))

    fig, ax = plt.subplots(figsize=(12, 4))

    for cls, label, color, lw, ls in lines:
        ax.plot(x, probs[:, cls], label=label, color=color,
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.axvline(FAULT_START, color='black', linestyle='--', linewidth=1.2,
               label=f'Fault inserted (t={FAULT_START})')

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Timestep (end index of window)')
    ax.set_ylabel('Softmax probability')
    ax.set_title(f'IDV14 — {run_id}  |  NOC + top-3 non-NOC class probabilities')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot IDV14 top-3 class probabilities for a model.')
    parser.add_argument('--model',       default='wavelet_kan',
                        help='Model key (default: wavelet_kan)')
    parser.add_argument('--results-dir', default='results_N200_tr160_v0_te40',
                        help='Results directory (default: results_N200_tr160_v0_te40)')
    parser.add_argument('--run',         default=None,
                        help='Plot only this Run_ID (e.g. IDV14_Run104); '
                             'omit to plot all IDV14 runs')
    args = parser.parse_args()

    preds_path = ROOT / args.results_dir / args.model / 'predictions.npz'
    if not preds_path.exists():
        print(f'ERROR: {preds_path} not found')
        sys.exit(1)

    data      = np.load(preds_path, allow_pickle=True)
    run_ids   = data['Run_ID']
    start_idx = data['start_idx']
    end_idx   = data['end_idx']
    y_prob    = data['y_prob']
    y_pred    = data['y_pred']
    y_true    = data['y_true']

    # Filter to IDV14
    idv14_mask = np.array(['IDV14' in str(r) for r in run_ids])
    if not idv14_mask.any():
        print('No IDV14 windows found in predictions.npz')
        sys.exit(1)

    unique_runs = np.unique(run_ids[idv14_mask])
    if args.run is not None:
        if args.run not in unique_runs:
            print(f'Run {args.run!r} not found. Available: {list(unique_runs[:5])} ...')
            sys.exit(1)
        unique_runs = [args.run]

    out_dir = ROOT / args.results_dir / args.model / 'idv14_top3_plots'
    out_dir.mkdir(exist_ok=True)

    print(f'Model:       {args.model}')
    print(f'Results dir: {args.results_dir}')
    print(f'IDV14 runs:  {len(unique_runs)}')
    print(f'Output dir:  {out_dir}')
    print()

    for run_id in unique_runs:
        mask = run_ids == run_id
        out_path = out_dir / f'{run_id}_top3_probs.png'
        plot_run(
            run_id   = run_id,
            end_idx  = end_idx[mask],
            y_prob   = y_prob[mask],
            y_pred   = y_pred[mask],
            y_true   = y_true[mask],
            out_path = out_path,
        )

    print(f'\nDone. {len(unique_runs)} plot(s) saved to {out_dir}/')


if __name__ == '__main__':
    main()
