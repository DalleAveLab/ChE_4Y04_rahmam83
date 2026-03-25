#!/usr/bin/env python3
# Smoke test for fault detection and diagnosis timing metrics.
# Produces 40 per-trial probability plots with detection/diagnosis markers,
# plus a timing_summary.txt whose averages must match model_scores_and_alarm_metrics.txt.
# Usage: python tests/smoke_test_timing.py

# ── USER CONFIG ──────────────────────────────────────────
MODEL       = 'wavelet_kan'       # efficient_kan | fourier_kan | wavelet_kan | fast_kan | mlp | cnn | rnn | lstm
FAULT_K     = 1                   # IDV number (1–28, excluding 6)
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

# Thresholds — must match evaluate.py:83-84 exactly
FAULT_START          = 600
DETECTION_THRESHOLD  = 0.10   # P(NOC) < this  → detected
DIAGNOSIS_CONFIDENCE = 0.90   # max P(non-NOC) > this → diagnosed


def compute_trial_timing(p_noc, p_all, x):
    """
    Given arrays for a single run (already post-fault-start filtered and sorted):
      p_noc : (n_post_windows,)
      p_all : (n_post_windows, n_classes)
      x     : end_idx values (n_post_windows,)
    Returns (fdet_time, fdiag_time) — both relative to FAULT_START, or None if not triggered.
    """
    # FDetT
    det = np.where(p_noc < DETECTION_THRESHOLD)[0]
    fdet_time = int(x[det[0]]) - FAULT_START if len(det) else None

    # FDiagT
    max_fault_prob = p_all[:, 1:].max(axis=1)
    diag = np.where(max_fault_prob > DIAGNOSIS_CONFIDENCE)[0]
    fdiag_time = int(x[diag[0]]) - FAULT_START if len(diag) else None

    return fdet_time, fdiag_time


def plot_trial(run_id, x_full, p_noc_full, p_fault_full,
               fdet_time, fdiag_time, fault_k, model, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(x_full, p_noc_full,   color='steelblue',  linewidth=1.0, label='P(NOC)')
    ax.plot(x_full, p_fault_full, color='darkorange',  linewidth=1.0, label=f'P(IDV{fault_k})')

    # Fault insertion line
    ax.axvline(x=FAULT_START, color='red', linestyle='--', linewidth=1.2,
               label=f'Fault inserted (t={FAULT_START})')
    ax.text(FAULT_START + 5, 0.92, f't={FAULT_START}', color='red', fontsize=7, va='top')

    # Detection line
    if fdet_time is not None:
        det_x = FAULT_START + fdet_time
        ax.axvline(x=det_x, color='green', linestyle='--', linewidth=1.2,
                   label=f'FDetT={fdet_time}')
        ax.text(det_x + 5, 0.82, f'FDetT={fdet_time}', color='green', fontsize=7, va='top')
    else:
        ax.text(0.62, 0.88, 'Not detected', color='green', fontsize=7,
                transform=ax.transAxes, va='top')

    # Diagnosis line
    if fdiag_time is not None:
        diag_x = FAULT_START + fdiag_time
        ax.axvline(x=diag_x, color='purple', linestyle='--', linewidth=1.2,
                   label=f'FDiagT={fdiag_time}')
        ax.text(diag_x + 5, 0.72, f'FDiagT={fdiag_time}', color='purple', fontsize=7, va='top')
    else:
        ax.text(0.62, 0.78, 'Not diagnosed', color='purple', fontsize=7,
                transform=ax.transAxes, va='top')

    ax.set_xlim(x_full[0], x_full[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Timestep (end of window)', fontsize=10)
    ax.set_ylabel('Softmax probability', fontsize=10)
    ax.set_title(f'{model} — IDV{fault_k} — {run_id}', fontsize=11)
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    config      = load_config(CONFIG_PATH)
    results_dir = Path(config.results_dir)
    preds_path  = results_dir / MODEL / 'predictions.npz'

    if not preds_path.exists():
        print(f"ERROR: {preds_path} not found.")
        sys.exit(1)

    print(f"Loading {preds_path} ...")
    data      = np.load(preds_path, allow_pickle=True)
    y_prob    = data['y_prob']
    run_ids   = data['Run_ID']
    start_idx = data['start_idx']
    end_idx   = data['end_idx']

    prefix    = f'IDV{FAULT_K}_'
    all_runs  = sorted(set(rid for rid in run_ids if rid.startswith(prefix)))
    if not all_runs:
        print(f"ERROR: No runs found for IDV{FAULT_K} in {preds_path}")
        sys.exit(1)

    # Output directory
    out_dir = Path(__file__).parent / f'smoke_test_IDV{FAULT_K}_{MODEL}'
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []  # (run_id, fdet_time, fdiag_time)

    for i, run_id in enumerate(all_runs, 1):
        mask  = run_ids == run_id
        order = np.argsort(start_idx[mask])

        p_noc_all   = y_prob[mask, 0][order]
        p_fault_all = y_prob[mask, FAULT_K][order]
        p_all_full  = y_prob[mask][order]
        x_full      = end_idx[mask][order]

        # Post-fault slice (same as evaluate.py:119-122)
        post        = x_full >= FAULT_START
        fdet_time, fdiag_time = compute_trial_timing(
            p_noc_all[post], p_all_full[post], x_full[post]
        )

        records.append((run_id, fdet_time, fdiag_time))

        out_path = out_dir / f'trial_{run_id}.png'
        plot_trial(run_id, x_full, p_noc_all, p_fault_all,
                   fdet_time, fdiag_time, FAULT_K, MODEL, out_path)
        print(f"  [{i:2d}/{len(all_runs)}] {run_id}  FDetT={fdet_time}  FDiagT={fdiag_time}  -> {out_path.name}")

    # ── Summary statistics (same formula as evaluate.py:152-158) ──
    fdet_vals  = [t for _, t, _ in records if t is not None]
    fdiag_vals = [t for _, _, t in records if t is not None]

    fdet_mean  = float(np.mean(fdet_vals))  if fdet_vals  else None
    fdet_std   = float(np.std(fdet_vals))   if fdet_vals  else None
    fdiag_mean = float(np.mean(fdiag_vals)) if fdiag_vals else None
    fdiag_std  = float(np.std(fdiag_vals))  if fdiag_vals else None

    n_total    = len(records)
    n_det      = len(fdet_vals)
    n_diag     = len(fdiag_vals)

    # ── Write timing_summary.txt ──
    txt_path = out_dir / 'timing_summary.txt'
    with open(txt_path, 'w') as f:
        f.write(f"Smoke Test Timing — MODEL={MODEL}  FAULT=IDV{FAULT_K}\n")
        f.write(f"Detection threshold : P(NOC) < {DETECTION_THRESHOLD}\n")
        f.write(f"Diagnosis threshold : max P(non-NOC) > {DIAGNOSIS_CONFIDENCE}\n")
        f.write(f"Fault insertion     : t={FAULT_START}\n")
        f.write("-" * 52 + "\n")
        f.write(f"{'Run_ID':<25}  {'FDetT':>7}  {'FDiagT':>7}\n")
        f.write("-" * 52 + "\n")
        for run_id, fdet, fdiag in records:
            fdet_str  = str(fdet)  if fdet  is not None else 'None'
            fdiag_str = str(fdiag) if fdiag is not None else 'None'
            f.write(f"{run_id:<25}  {fdet_str:>7}  {fdiag_str:>7}\n")
        f.write("-" * 52 + "\n")
        if fdet_mean is not None:
            f.write(f"Detected  : {n_det}/{n_total}    Mean FDetT  : {fdet_mean:.1f} +/- {fdet_std:.1f}\n")
        else:
            f.write(f"Detected  : 0/{n_total}    Mean FDetT  : N/A\n")
        if fdiag_mean is not None:
            f.write(f"Diagnosed : {n_diag}/{n_total}    Mean FDiagT : {fdiag_mean:.1f} +/- {fdiag_std:.1f}\n")
        else:
            f.write(f"Diagnosed : 0/{n_total}    Mean FDiagT : N/A\n")
        f.write("-" * 52 + "\n")
        f.write("Compare against model_scores_and_alarm_metrics.txt:\n")
        f.write(f"  FDetT  row IDV{FAULT_K}, col {MODEL}\n")
        f.write(f"  FDiagT row IDV{FAULT_K}, col {MODEL}\n")
        f.write("  (values should match mean+/-std and detected/total exactly)\n")

    print(f"\nSummary written to: {txt_path}")
    if fdet_mean is not None:
        print(f"  FDetT  : {fdet_mean:.1f} +/- {fdet_std:.1f}  ({n_det}/{n_total} detected)")
    else:
        print(f"  FDetT  : N/A  (0/{n_total} detected)")
    if fdiag_mean is not None:
        print(f"  FDiagT : {fdiag_mean:.1f} +/- {fdiag_std:.1f}  ({n_diag}/{n_total} diagnosed)")
    else:
        print(f"  FDiagT : N/A  (0/{n_total} diagnosed)")


if __name__ == '__main__':
    main()
