#!/usr/bin/env python3
# Time-series probability plots for TEP fault detection evaluation.
# For each fault class, averages P(NOC) and P(IDVk) across all test runs
# and saves a PNG showing mean ± 1 std with fault insertion marker.

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_time_series_per_fault(
    y_prob:    np.ndarray,
    y_true:    np.ndarray,
    run_ids:   np.ndarray,
    start_idx: np.ndarray,
    end_idx:   np.ndarray,
    variant:   str,
    out_dir:   Path,
    fault_start: int = 600,
) -> None:
    """
    For each fault class present in run_ids, average P(healthy) and P(IDV#)
    across all runs of that class and save a time-series plot as PNG.

    Expected pattern: P(healthy) dominates before fault_start, then collapses
    as P(IDV#) rises.
    """
    # Parse fault classes from Run_ID strings (format: 'IDV{k}_Run{n}')
    unique_run_ids = np.unique(run_ids)
    fault_classes = sorted(set(
        int(rid.split('_')[0][3:]) for rid in unique_run_ids
    ))

    for k in fault_classes:
        prefix = f'IDV{k}_'
        runs_for_k = [rid for rid in unique_run_ids if rid.startswith(prefix)]
        if not runs_for_k:
            continue

        healthy_curves = []
        fault_curves   = []
        x_axis         = None

        for run_id in sorted(runs_for_k):
            mask = run_ids == run_id
            # Sort windows within this run by start position
            order = np.argsort(start_idx[mask])
            p_healthy = y_prob[mask, 0][order]
            p_fault   = y_prob[mask, k][order]
            x         = end_idx[mask][order]
            healthy_curves.append(p_healthy)
            fault_curves.append(p_fault)
            if x_axis is None:
                x_axis = x

        # Stack → (n_runs, n_windows_per_run); trim to minimum length across runs
        min_len = min(len(c) for c in healthy_curves)
        healthy_mat = np.stack([c[:min_len] for c in healthy_curves])
        fault_mat   = np.stack([c[:min_len] for c in fault_curves])
        x_axis      = x_axis[:min_len]

        mean_healthy = healthy_mat.mean(axis=0)
        std_healthy  = healthy_mat.std(axis=0)
        mean_fault   = fault_mat.mean(axis=0)
        std_fault    = fault_mat.std(axis=0)

        n_runs = len(runs_for_k)
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(x_axis, mean_healthy, color='steelblue',  label='P(NOC)',    linewidth=1.0)
        ax.fill_between(x_axis,
                        mean_healthy - std_healthy,
                        mean_healthy + std_healthy,
                        alpha=0.2, color='steelblue')

        ax.plot(x_axis, mean_fault,   color='darkorange', label=f'P(IDV{k})',    linewidth=1.0)
        ax.fill_between(x_axis,
                        mean_fault - std_fault,
                        mean_fault + std_fault,
                        alpha=0.2, color='darkorange')

        ax.axvline(x=fault_start, color='red', linestyle='--', linewidth=1.2,
                   label=f'Fault inserted (t={fault_start})')

        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Timestep (end of window)', fontsize=10)
        ax.set_ylabel('Softmax probability', fontsize=10)
        ax.set_title(
            f'{variant} — IDV{k}: Mean probability over {n_runs} test runs (±1 std)',
            fontsize=11
        )
        ax.legend(fontsize=9, loc='center right')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f'time_series_IDV{k}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    from configs.config_loader import load_config
    from scripts.evaluate import ALL_VARIANTS

    parser = argparse.ArgumentParser(description='Generate time-series probability plots for TEP variants')
    parser.add_argument('--model', type=str, default=None,
                        choices=ALL_VARIANTS,
                        help='Plot a single variant (default: all)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config.results_dir)
    variants = [args.model] if args.model else ALL_VARIANTS

    for variant in variants:
        preds_path = results_dir / variant / 'predictions.npz'
        if not preds_path.exists():
            print(f"  WARNING: {preds_path} not found — skipping {variant}")
            continue
        data = np.load(preds_path, allow_pickle=True)
        if not all(k in data for k in ('y_prob', 'y_true', 'Run_ID', 'start_idx', 'end_idx')):
            print(f"  WARNING: {variant} predictions.npz missing metadata — skipping")
            continue
        print(f"\n  Plotting: {variant}")
        plot_time_series_per_fault(
            data['y_prob'], data['y_true'],
            data['Run_ID'], data['start_idx'], data['end_idx'],
            variant, results_dir / variant,
        )


if __name__ == '__main__':
    main()
