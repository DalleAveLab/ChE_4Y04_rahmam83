#!/usr/bin/env python3
"""
Offline evaluation of tuned KAN variants on TEP fault detection.

Loads saved predictions.npz files (no model rerun) and computes metrics.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model efficient_kan
    python scripts/evaluate.py --config configs/config.yaml
"""

import sys
import json
import argparse
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.config_loader import load_config

ALL_VARIANTS = [
    'efficient_kan', 'fourier_kan', 'wavelet_kan', 'fast_kan',
    'mlp', 'cnn', 'rnn', 'lstm',
]


def plot_loss_curve(epoch_losses: list, variant: str, out_path: Path):
    """Save a training loss curve PNG for one variant."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'Training Loss — {variant}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list, variant: str, out_path: Path):
    """Save a row-normalized confusion matrix heatmap as PNG."""
    n = len(classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.55)))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall (row-normalized)')

    tick_labels = [f'C{c}' for c in classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'Confusion Matrix (row-normalized) — {variant}', fontsize=11)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


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

        ax.plot(x_axis, mean_healthy, color='steelblue',  label='P(healthy)',    linewidth=1.0)
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


def evaluate_variant(results_dir: Path, variant: str) -> dict | None:
    """
    Load predictions.npz for a variant and compute metrics.

    Returns None if the predictions file doesn't exist yet.
    """
    preds_path = results_dir / variant / 'predictions.npz'
    if not preds_path.exists():
        print(f"  WARNING: {preds_path} not found — skipping {variant}")
        return None

    data = np.load(preds_path, allow_pickle=True)
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_prob = data['y_prob'] if 'y_prob' in data else None  # (n_windows, n_classes)

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class F1
    classes = sorted(set(y_true) | set(y_pred))
    per_class_f1_arr = f1_score(y_true, y_pred, labels=classes,
                                average=None, zero_division=0)
    per_class_f1 = {int(c): float(f) for c, f in zip(classes, per_class_f1_arr)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    ALARM_THRESHOLD = 0.90  # fault probability threshold for raising an alarm

    # Mean prediction confidence and alarm stats (when y_prob is available)
    if y_prob is not None:
        # Confidence = softmax probability of the predicted class
        confidence = y_prob[np.arange(len(y_pred)), y_pred]
        correct_mask = y_pred == y_true
        mean_conf_correct = float(confidence[correct_mask].mean()) if correct_mask.any() else None
        mean_conf_wrong   = float(confidence[~correct_mask].mean()) if (~correct_mask).any() else None

        # Alarm: raised when fault probability (1 - P(class 0)) exceeds threshold
        fault_prob  = 1.0 - y_prob[:, 0]
        alarm       = fault_prob > ALARM_THRESHOLD
        healthy_mask = y_true == 0
        fault_mask   = y_true != 0
        false_alarm_rate    = float(alarm[healthy_mask].mean()) if healthy_mask.any() else None
        detection_rate      = float(alarm[fault_mask].mean())   if fault_mask.any()   else None
        correct_normal_rate = (1.0 - false_alarm_rate) if false_alarm_rate is not None else None
        miss_rate           = (1.0 - detection_rate)   if detection_rate   is not None else None

        # Per-class FDR: detection rate for each individual fault class
        fault_classes_list = sorted(c for c in set(y_true) if c != 0)
        per_class_detection_rate = {
            int(k): float(alarm[y_true == k].mean()) if (y_true == k).any() else None
            for k in fault_classes_list
        }

        # Top-2 margin: gap between highest and second-highest class probability.
        # A small margin (e.g. 35% vs 34%) means the model is uncertain even when
        # it produces a prediction. Full distribution is in y_prob for deeper inspection.
        sorted_probs = np.sort(y_prob, axis=1)          # ascending per row
        top2_margin  = sorted_probs[:, -1] - sorted_probs[:, -2]  # (n_windows,)
        mean_top2_margin         = float(top2_margin.mean())
        frac_ambiguous           = float((top2_margin < 0.10).mean())  # < 10pp gap
    else:
        confidence = None
        mean_conf_correct = None
        mean_conf_wrong   = None
        false_alarm_rate         = None
        detection_rate           = None
        correct_normal_rate      = None
        miss_rate                = None
        per_class_detection_rate = {}
        mean_top2_margin         = None
        frac_ambiguous           = None

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm.tolist(),
        'classes': [int(c) for c in classes],
        'mean_conf_correct': mean_conf_correct,
        'mean_conf_wrong':   mean_conf_wrong,
        'alarm_threshold':   ALARM_THRESHOLD,
        'false_alarm_rate':    false_alarm_rate,
        'detection_rate':            detection_rate,
        'correct_normal_rate':       correct_normal_rate,
        'miss_rate':                 miss_rate,
        'per_class_detection_rate':  per_class_detection_rate,
        'mean_top2_margin':  mean_top2_margin,
        'frac_ambiguous':    frac_ambiguous,
    }

    # ------------------------------------------------------------------
    # Load additional saved data (best_params.json, metrics.json)
    # ------------------------------------------------------------------
    variant_dir = results_dir / variant
    plot_confusion_matrix(cm, classes, variant, variant_dir / 'confusion_matrix.png')

    # Time-series probability plots (requires Run_ID metadata)
    if all(k in data for k in ('Run_ID', 'start_idx', 'end_idx')) and y_prob is not None:
        plot_time_series_per_fault(
            y_prob, y_true,
            data['Run_ID'], data['start_idx'], data['end_idx'],
            variant, variant_dir,
        )

    # Best params from tuning
    params_path = variant_dir / 'best_params.json'
    if params_path.exists():
        with open(params_path, 'r') as f:
            metrics['best_params'] = json.load(f)
    else:
        metrics['best_params'] = {}

    # Tuning metrics (val_accuracy, best_trial, n_trials) and loss curve
    tuning_metrics_path = variant_dir / 'metrics.json'
    if tuning_metrics_path.exists():
        with open(tuning_metrics_path, 'r') as f:
            tuning_metrics = json.load(f)
        metrics['val_accuracy'] = tuning_metrics.get('val_accuracy')
        metrics['n_trials'] = tuning_metrics.get('n_trials')
        metrics['best_trial'] = tuning_metrics.get('best_trial')

        loss_curve = (tuning_metrics.get('train_loss_curve')
                      or tuning_metrics.get('train_elbo_curve'))
        if loss_curve:
            plot_loss_curve(loss_curve, variant, variant_dir / 'loss_curve.png')
    else:
        metrics['val_accuracy'] = None
        metrics['n_trials'] = None
        metrics['best_trial'] = None

    # Save eval_metrics.json
    out_path = variant_dir / 'eval_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size:,} bytes)")

    return metrics


# ======================================================================
# Display helpers
# ======================================================================

def format_params_short(params: dict) -> str:
    """Format best_params dict into a compact string for table display."""
    if not params:
        return 'N/A'

    parts = []
    # Universal params first
    if 'hidden_dim' in params:
        parts.append(f"hd={params['hidden_dim']}")
    if 'hidden_layers' in params:
        parts.append(f"hl={params['hidden_layers']}")
    if 'lr' in params:
        parts.append(f"lr={params['lr']:.1e}")

    # Variant-specific params (everything else)
    skip_keys = {'hidden_dim', 'hidden_layers', 'lr'}
    for k, v in params.items():
        if k not in skip_keys:
            short_k = k.replace('_', '')
            if isinstance(v, float):
                parts.append(f"{short_k}={v:.2g}")
            else:
                parts.append(f"{short_k}={v}")

    return ', '.join(parts)


def print_comparison_table(results: dict):
    """Print a formatted comparison table across all evaluated variants."""
    if not results:
        print("\n  No variants have been evaluated yet.")
        return

    # ==================================================================
    # Section 1: Model Comparison (main table with params)
    # ==================================================================
    print(f"\n{'='*120}")
    print("  Model Comparison")
    print(f"{'='*120}")

    header = (f"  {'Model':<18s} {'Accuracy':>10s} {'Macro F1':>10s} "
              f"{'Val Acc':>10s} {'Gap':>8s} {'Conf(Correct)':>14s} {'Conf(Wrong)':>12s} {'Best Params'}")
    print(f"\n{header}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*14} {'-'*12} {'-'*55}")

    best_model = None
    best_acc = -1.0

    for variant, m in results.items():
        acc = m['accuracy']
        f1 = m['macro_f1']
        val_acc = m.get('val_accuracy')
        params_str = format_params_short(m.get('best_params', {}))

        # Val-test gap
        if val_acc is not None:
            gap = val_acc - acc
            gap_str = f"{gap:+.4f}"
            val_str = f"{val_acc:.4f}"
        else:
            gap_str = "N/A"
            val_str = "N/A"

        cc = m.get('mean_conf_correct')
        cw = m.get('mean_conf_wrong')
        cc_str = f"{cc:.4f}" if cc is not None else "N/A"
        cw_str = f"{cw:.4f}" if cw is not None else "N/A"

        print(f"  {variant:<18s} {acc:>10.4f} {f1:>10.4f} "
              f"{val_str:>10s} {gap_str:>8s} {cc_str:>14s} {cw_str:>12s} {params_str}")

        if acc > best_acc:
            best_acc = acc
            best_model = variant

    print(f"\n  Best model: {best_model} (accuracy = {best_acc:.4f})")

    # ==================================================================
    # Section 2: Tuning Summary
    # ==================================================================
    print(f"\n{'='*120}")
    print("  Tuning Summary")
    print(f"{'='*120}")

    print(f"\n  {'Model':<18s} {'Trials':>8s} {'Best Trial':>12s} "
          f"{'Val Acc':>10s} {'Test Acc':>10s} {'Overfit?':>10s}")
    print(f"  {'-'*18} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for variant, m in results.items():
        n_trials = m.get('n_trials')
        best_trial = m.get('best_trial')
        val_acc = m.get('val_accuracy')
        test_acc = m['accuracy']

        n_str = str(n_trials) if n_trials is not None else 'N/A'
        bt_str = str(best_trial) if best_trial is not None else 'N/A'
        val_str = f"{val_acc:.4f}" if val_acc is not None else 'N/A'

        # Flag potential overfitting (val-test gap > 2%)
        if val_acc is not None:
            gap = val_acc - test_acc
            overfit_str = "Yes" if gap > 0.02 else "No"
        else:
            overfit_str = "N/A"

        print(f"  {variant:<18s} {n_str:>8s} {bt_str:>12s} "
              f"{val_str:>10s} {test_acc:>10.4f} {overfit_str:>10s}")

    # ==================================================================
    # Section 3: Per-Class F1 Comparison
    # ==================================================================
    print(f"\n{'='*120}")
    print("  Per-Class F1 Comparison")
    print(f"{'='*120}")

    # Collect all classes across all models
    all_classes = set()
    for m in results.values():
        all_classes.update(m.get('per_class_f1', {}).keys())
    all_classes = sorted(all_classes)

    if all_classes:
        # Header
        class_labels = [f"C{c}" for c in all_classes]
        header = f"  {'Model':<18s} " + " ".join(f"{cl:>6s}" for cl in class_labels)
        print(f"\n{header}")
        print(f"  {'-'*18} " + " ".join("-" * 6 for _ in class_labels))

        # Per-model rows
        for variant, m in results.items():
            pf1 = m.get('per_class_f1', {})
            row = f"  {variant:<18s} "
            row += " ".join(f"{pf1.get(c, 0.0):>6.3f}" for c in all_classes)
            print(row)

        # Best model per class
        print(f"  {'-'*18} " + " ".join("-" * 6 for _ in class_labels))
        best_row = f"  {'Best model':<18s} "
        for c in all_classes:
            scores = {v: m.get('per_class_f1', {}).get(c, 0.0)
                      for v, m in results.items()}
            best_v = max(scores, key=scores.get)
            abbrev = {'efficient_kan': 'EFF', 'fourier_kan': 'FOU',
                      'wavelet_kan': 'WAV', 'fast_kan': 'FST'}
            best_row += f"{abbrev.get(best_v, best_v[:3]):>6s} "
        print(best_row)

        # Highlight weakest classes per model
        print(f"\n  Weakest classes per model (F1 < 0.85):")
        for variant, m in results.items():
            pf1 = m.get('per_class_f1', {})
            weak = {c: f for c, f in pf1.items() if f < 0.85}
            if weak:
                weak_sorted = sorted(weak.items(), key=lambda x: x[1])
                weak_str = ", ".join(f"C{c}={f:.3f}" for c, f in weak_sorted)
                print(f"    {variant:<18s}: {weak_str}")
            else:
                print(f"    {variant:<18s}: None (all classes >= 0.85)")

    # ==================================================================
    # Section 4: Confusion Matrix — Top Misclassification Pairs
    # ==================================================================
    print(f"\n{'='*120}")
    print("  Confusion Matrix — Top Misclassification Pairs")
    print(f"{'='*120}")

    for variant, m in results.items():
        cm = np.array(m['confusion_matrix'])
        classes = m.get('classes', list(range(cm.shape[0])))

        # Zero out diagonal to find off-diagonal hotspots
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        total_misclassified = cm_off.sum()

        if total_misclassified == 0:
            print(f"\n  {variant}: Perfect classification (no errors)")
            continue

        # Find top-5 misclassification pairs
        flat_indices = np.argsort(cm_off.ravel())[::-1][:5]
        top_pairs = []
        for idx in flat_indices:
            i, j = divmod(idx, cm_off.shape[1])
            count = cm_off[i, j]
            if count > 0:
                top_pairs.append((classes[i], classes[j], int(count)))

        print(f"\n  {variant} (total misclassified: {int(total_misclassified):,}):")
        print(f"    {'True':>6s} -> {'Pred':>6s}   {'Count':>6s}   {'% of Errors':>12s}")
        print(f"    {'-'*6}    {'-'*6}   {'-'*6}   {'-'*12}")
        for true_c, pred_c, count in top_pairs:
            pct = 100.0 * count / total_misclassified
            print(f"    C{true_c:>4d} -> C{pred_c:>4d}   {count:>6d}   {pct:>11.1f}%")

    # ==================================================================
    # Section 5: Alarm Analysis (90% fault-probability threshold)
    # ==================================================================
    has_alarm_data = any(
        m.get('false_alarm_rate') is not None or m.get('detection_rate') is not None
        for m in results.values()
    )
    if has_alarm_data:
        threshold = next(
            (m['alarm_threshold'] for m in results.values() if 'alarm_threshold' in m),
            0.90
        )
        print(f"\n{'='*120}")
        print(f"  Alarm Analysis  (threshold: fault_prob = 1 - P(class 0) > {threshold:.0%})")
        print(f"{'='*120}")
        print(f"\n  {'Model':<18s} {'False Alarm Rate':>18s} {'Correct Normal Rate':>20s} "
              f"{'Detection Rate':>16s} {'Miss Rate':>12s} "
              f"{'Mean Top-2 Margin':>20s} {'Ambiguous (<10pp)':>18s}")
        print(f"  {'-'*18} {'-'*18} {'-'*20} {'-'*16} {'-'*12} {'-'*20} {'-'*18}")
        print(f"  {'':18s} {'(healthy → alarm)':>18s} {'(healthy → no alarm)':>20s} "
              f"{'(fault → alarm)':>16s} {'(fault → no alarm)':>12s} "
              f"{'(top1 - top2 prob)':>20s} {'(% windows)':>18s}")
        print(f"  {'-'*18} {'-'*18} {'-'*20} {'-'*16} {'-'*12} {'-'*20} {'-'*18}")
        sanity_far = sanity_dr = None
        for variant, m in results.items():
            far    = m.get('false_alarm_rate')
            cnr    = m.get('correct_normal_rate')
            dr     = m.get('detection_rate')
            mr     = m.get('miss_rate')
            margin = m.get('mean_top2_margin')
            ambig  = m.get('frac_ambiguous')
            far_str    = f"{far:.4f} ({far*100:.1f}%)"       if far    is not None else "N/A"
            cnr_str    = f"{cnr:.4f} ({cnr*100:.1f}%)"       if cnr    is not None else "N/A"
            dr_str     = f"{dr:.4f} ({dr*100:.1f}%)"         if dr     is not None else "N/A"
            mr_str     = f"{mr:.4f} ({mr*100:.1f}%)"         if mr     is not None else "N/A"
            margin_str = f"{margin:.4f} ({margin*100:.1f}pp)" if margin is not None else "N/A"
            ambig_str  = f"{ambig*100:.1f}%"                  if ambig  is not None else "N/A"
            print(f"  {variant:<18s} {far_str:>18s} {cnr_str:>20s} "
                  f"{dr_str:>16s} {mr_str:>12s} "
                  f"{margin_str:>20s} {ambig_str:>18s}")
            if sanity_far is None and far is not None and cnr is not None:
                sanity_far = far + cnr
            if sanity_dr is None and dr is not None and mr is not None:
                sanity_dr = dr + mr
        print(f"\n  Sanity check — FAR + CNR = {sanity_far:.6f}  |  FDR + MR = {sanity_dr:.6f}"
              f"  (both should equal 1.000000)"
              if sanity_far is not None and sanity_dr is not None else "")

        # Per-class FDR table (mirrors per-class F1 layout)
        all_fault_classes = set()
        for m in results.values():
            all_fault_classes.update(m.get('per_class_detection_rate', {}).keys())
        all_fault_classes = sorted(all_fault_classes)

        if all_fault_classes:
            print(f"\n  Per-Class Detection Rate  (fault → alarm, threshold {threshold:.0%})")
            class_labels = [f"IDV{c}" for c in all_fault_classes]
            header = f"  {'Model':<18s} " + " ".join(f"{cl:>7s}" for cl in class_labels)
            print(f"\n{header}")
            print(f"  {'-'*18} " + " ".join("-" * 7 for _ in class_labels))
            for variant, m in results.items():
                pcdr = m.get('per_class_detection_rate', {})
                row = f"  {variant:<18s} "
                row += " ".join(
                    f"{pcdr[c]:>7.3f}" if pcdr.get(c) is not None else f"{'N/A':>7s}"
                    for c in all_fault_classes
                )
                print(row)
            # Highlight worst-detected fault classes per model
            print(f"\n  Weakest fault classes per model (FDR < 0.85):")
            for variant, m in results.items():
                pcdr = m.get('per_class_detection_rate', {})
                weak = {c: v for c, v in pcdr.items() if v is not None and v < 0.85}
                if weak:
                    weak_sorted = sorted(weak.items(), key=lambda x: x[1])
                    weak_str = ", ".join(f"IDV{c}={v:.3f}" for c, v in weak_sorted)
                    print(f"    {variant:<18s}: {weak_str}")
                else:
                    print(f"    {variant:<18s}: None (all fault classes >= 0.85)")

    print(f"\n{'='*120}")
    print("  Evaluation complete.")
    print(f"{'='*120}")


def save_alarm_metrics(results: dict, results_dir: Path):
    """Write the alarm analysis tables to alarm_metrics.txt in the results directory."""
    has_alarm_data = any(
        m.get('false_alarm_rate') is not None or m.get('detection_rate') is not None
        for m in results.values()
    )
    if not has_alarm_data:
        return

    lines = []
    threshold = next(
        (m['alarm_threshold'] for m in results.values() if 'alarm_threshold' in m),
        0.90
    )

    lines.append("=" * 120)
    lines.append(f"  Alarm Analysis  (threshold: fault_prob = 1 - P(class 0) > {threshold:.0%})")
    lines.append("=" * 120)
    lines.append(
        f"\n  {'Model':<18s} {'False Alarm Rate':>18s} {'Correct Normal Rate':>20s} "
        f"{'Detection Rate':>16s} {'Miss Rate':>12s} "
        f"{'Mean Top-2 Margin':>20s} {'Ambiguous (<10pp)':>18s}"
    )
    lines.append(
        f"  {'-'*18} {'-'*18} {'-'*20} {'-'*16} {'-'*12} {'-'*20} {'-'*18}"
    )
    lines.append(
        f"  {'':18s} {'(healthy → alarm)':>18s} {'(healthy → no alarm)':>20s} "
        f"{'(fault → alarm)':>16s} {'(fault → no alarm)':>12s} "
        f"{'(top1 - top2 prob)':>20s} {'(% windows)':>18s}"
    )
    lines.append(
        f"  {'-'*18} {'-'*18} {'-'*20} {'-'*16} {'-'*12} {'-'*20} {'-'*18}"
    )

    sanity_far = sanity_dr = None
    for variant, m in results.items():
        far    = m.get('false_alarm_rate')
        cnr    = m.get('correct_normal_rate')
        dr     = m.get('detection_rate')
        mr     = m.get('miss_rate')
        margin = m.get('mean_top2_margin')
        ambig  = m.get('frac_ambiguous')
        far_str    = f"{far:.4f} ({far*100:.1f}%)"        if far    is not None else "N/A"
        cnr_str    = f"{cnr:.4f} ({cnr*100:.1f}%)"        if cnr    is not None else "N/A"
        dr_str     = f"{dr:.4f} ({dr*100:.1f}%)"          if dr     is not None else "N/A"
        mr_str     = f"{mr:.4f} ({mr*100:.1f}%)"          if mr     is not None else "N/A"
        margin_str = f"{margin:.4f} ({margin*100:.1f}pp)"  if margin is not None else "N/A"
        ambig_str  = f"{ambig*100:.1f}%"                   if ambig  is not None else "N/A"
        lines.append(
            f"  {variant:<18s} {far_str:>18s} {cnr_str:>20s} "
            f"{dr_str:>16s} {mr_str:>12s} "
            f"{margin_str:>20s} {ambig_str:>18s}"
        )
        if sanity_far is None and far is not None and cnr is not None:
            sanity_far = far + cnr
        if sanity_dr is None and dr is not None and mr is not None:
            sanity_dr = dr + mr

    if sanity_far is not None and sanity_dr is not None:
        lines.append(
            f"\n  Sanity check — FAR + CNR = {sanity_far:.6f}  |  FDR + MR = {sanity_dr:.6f}"
            f"  (both should equal 1.000000)"
        )

    # Per-class FDR table
    all_fault_classes = set()
    for m in results.values():
        all_fault_classes.update(m.get('per_class_detection_rate', {}).keys())
    all_fault_classes = sorted(all_fault_classes)

    if all_fault_classes:
        lines.append(f"\n  Per-Class Detection Rate  (fault → alarm, threshold {threshold:.0%})")
        class_labels = [f"IDV{c}" for c in all_fault_classes]
        lines.append(f"\n  {'Model':<18s} " + " ".join(f"{cl:>7s}" for cl in class_labels))
        lines.append(f"  {'-'*18} " + " ".join("-" * 7 for _ in class_labels))
        for variant, m in results.items():
            pcdr = m.get('per_class_detection_rate', {})
            row = f"  {variant:<18s} "
            row += " ".join(
                f"{pcdr[c]:>7.3f}" if pcdr.get(c) is not None else f"{'N/A':>7s}"
                for c in all_fault_classes
            )
            lines.append(row)

        lines.append(f"\n  Weakest fault classes per model (FDR < 0.85):")
        for variant, m in results.items():
            pcdr = m.get('per_class_detection_rate', {})
            weak = {c: v for c, v in pcdr.items() if v is not None and v < 0.85}
            if weak:
                weak_sorted = sorted(weak.items(), key=lambda x: x[1])
                weak_str = ", ".join(f"IDV{c}={v:.3f}" for c, v in weak_sorted)
                lines.append(f"    {variant:<18s}: {weak_str}")
            else:
                lines.append(f"    {variant:<18s}: None (all fault classes >= 0.85)")

    lines.append("=" * 120)

    out_path = results_dir / 'alarm_metrics.txt'
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved alarm metrics: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate tuned KAN variants')
    parser.add_argument('--model', type=str, default=None,
                        choices=ALL_VARIANTS,
                        help='Evaluate a single variant (default: all)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config.results_dir)

    variants = [args.model] if args.model else ALL_VARIANTS

    print("=" * 120)
    print("  KAN Evaluation — TEP Fault Detection")
    print("=" * 120)

    results = {}
    for variant in variants:
        print(f"\n  Evaluating: {variant}")
        print(f"  {'-'*40}")
        metrics = evaluate_variant(results_dir, variant)
        if metrics is not None:
            results[variant] = metrics

    print_comparison_table(results)
    save_alarm_metrics(results, results_dir)


if __name__ == '__main__':
    main()