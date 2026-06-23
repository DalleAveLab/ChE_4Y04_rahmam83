#!/usr/bin/env python3
# Zoomed time-series plot around the fault insertion point for a given IDV and model.
# Shows mean P(NOC), P(true fault), and P(top misclassified class) across all test runs
# in a narrow window around t=FAULT_START, to reveal transient misclassification spikes.
# Usage: python tests/smoke_test_transition.py

# ── USER CONFIG ──────────────────────────────────────────
MODEL       = 'wavelet_kan'   # efficient_kan | fourier_kan | wavelet_kan | fast_kan | mlp | cnn | rnn | lstm
FAULT_K     = 14              # IDV number (1–28, excluding 6)
ZOOM_PRE    = 20              # timesteps before fault insertion to show
ZOOM_POST   = 40              # timesteps after fault insertion to show
CONFIG_PATH = 'configs/config.yaml'
# ─────────────────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.config_loader import load_config

FAULT_START = 600


def main():
    config      = load_config(CONFIG_PATH)
    results_dir = Path(config.results_dir)
    preds_path  = results_dir / MODEL / 'predictions.npz'

    if not preds_path.exists():
        print(f"ERROR: {preds_path} not found.")
        sys.exit(1)

    print(f"Loading {preds_path} ...")
    data      = np.load(preds_path, allow_pickle=True)
    y_prob    = data['y_prob']       # (n_total_windows, n_classes)
    y_pred    = data['y_pred']       # (n_total_windows,)
    y_true    = data['y_true']       # (n_total_windows,)
    run_ids   = data['Run_ID']
    start_idx = data['start_idx']
    end_idx   = data['end_idx']

    prefix   = f'IDV{FAULT_K}_'
    all_runs = sorted(set(rid for rid in run_ids if rid.startswith(prefix)))
    if not all_runs:
        print(f"ERROR: No runs found for IDV{FAULT_K}.")
        sys.exit(1)

    n_classes = y_prob.shape[1]
    x_min = FAULT_START - ZOOM_PRE
    x_max = FAULT_START + ZOOM_POST

    # Collect per-run probability curves in the zoom window
    # Shape will be (n_runs, n_timesteps_in_zoom, n_classes)
    prob_curves = []
    x_axis      = None

    for run_id in all_runs:
        mask  = run_ids == run_id
        order = np.argsort(start_idx[mask])
        x     = end_idx[mask][order]
        probs = y_prob[mask][order]          # (n_windows, n_classes)

        zoom = (x >= x_min) & (x <= x_max)
        if not zoom.any():
            continue

        prob_curves.append(probs[zoom])
        if x_axis is None:
            x_axis = x[zoom]

    if not prob_curves:
        print("ERROR: No windows found in the zoom range.")
        sys.exit(1)

    # Trim to same length in case of off-by-one across runs
    min_len     = min(len(c) for c in prob_curves)
    prob_mat    = np.stack([c[:min_len] for c in prob_curves])  # (n_runs, n_zoom, n_classes)
    x_axis      = x_axis[:min_len]

    mean_probs = prob_mat.mean(axis=0)   # (n_zoom, n_classes)
    std_probs  = prob_mat.std(axis=0)

    # Find top misclassified class (excluding NOC and true fault)
    # Look at post-fault windows only
    post_mask = x_axis >= FAULT_START
    if post_mask.any():
        mean_post = mean_probs[post_mask]                # (n_post, n_classes)
        exclude   = {0, FAULT_K}
        candidate_means = [
            (c, mean_post[:, c].max()) for c in range(n_classes) if c not in exclude
        ]
        top_other_class, top_other_val = max(candidate_means, key=lambda x: x[1])
    else:
        top_other_class = None

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    plt.rcParams['font.family'] = 'sans-serif'

    C_NOC    = '#4C8AC4'      # medium blue
    C_FAULT  = '#F07F2D'      # warm orange
    C_OTHER  = '#2EAA6E'      # green
    C_VLINE  = '#C0392B'      # deep red for fault insertion marker

    # P(NOC)
    ax.plot(x_axis, mean_probs[:, 0], color=C_NOC, linewidth=2.0, label='P(NOC)', zorder=3)
    ax.fill_between(x_axis,
                    mean_probs[:, 0] - std_probs[:, 0],
                    mean_probs[:, 0] + std_probs[:, 0],
                    alpha=0.18, color=C_NOC, zorder=2)

    # P(true fault)
    ax.plot(x_axis, mean_probs[:, FAULT_K], color=C_FAULT, linewidth=2.0,
            label=f'P(IDV{FAULT_K})', zorder=3)
    ax.fill_between(x_axis,
                    mean_probs[:, FAULT_K] - std_probs[:, FAULT_K],
                    mean_probs[:, FAULT_K] + std_probs[:, FAULT_K],
                    alpha=0.18, color=C_FAULT, zorder=2)

    # Top misclassified class
    if top_other_class is not None:
        ax.plot(x_axis, mean_probs[:, top_other_class], color=C_OTHER, linewidth=1.8,
                linestyle='-', label=f'P(IDV{top_other_class}) [top other]', zorder=3)
        ax.fill_between(x_axis,
                        mean_probs[:, top_other_class] - std_probs[:, top_other_class],
                        mean_probs[:, top_other_class] + std_probs[:, top_other_class],
                        alpha=0.13, color=C_OTHER, zorder=2)

    # Fault insertion line
    ax.axvline(x=FAULT_START, color=C_VLINE, linestyle='--', linewidth=1.4,
               label=f'Fault inserted (t={FAULT_START})', zorder=4)

    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Timestep (end of window)', fontsize=11)
    ax.set_ylabel('Softmax probability', fontsize=11)
    ax.set_title(
        f'{MODEL} — IDV{FAULT_K}: Transition zoom  '
        f'(t={x_axis[0]}–{x_axis[-1]}, mean ±1 std over {len(prob_curves)} runs)',
        fontsize=12
    )
    ax.legend(fontsize=10, loc='center right', framealpha=0.85, edgecolor='#cccccc')
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    out_dir  = Path(__file__).parent / f'smoke_test_IDV{FAULT_K}_{MODEL}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'transition_zoom_IDV{FAULT_K}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
