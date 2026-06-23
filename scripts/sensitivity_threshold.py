#!/usr/bin/env python3
# Alarm-threshold sensitivity analysis: re-evaluates a model's saved predictions
# at different P(NOC) detection thresholds, without retraining.
# Usage: python scripts/sensitivity_threshold.py
#        python scripts/sensitivity_threshold.py --model wavelet_kan,mlp,cnn
#        python scripts/sensitivity_threshold.py --model wavelet_kan --thresholds 0.05,0.075,0.10,0.125,0.15

import sys
import argparse
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import openpyxl

from configs.config_loader import load_config
from scripts.evaluate import (
    compute_alarm_metrics, compute_timing_metrics,
    _sheet_alarm_analysis, _sheet_per_class_fdr, _sheet_fault_detection_time,
    ALL_VARIANTS,
)

DEFAULT_THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15]
DEFAULT_MODELS = ['wavelet_kan', 'mlp', 'cnn']


def format_threshold_label(t: float) -> str:
    return f"{t * 100:g}%"


def run_threshold_sensitivity(results_dir: Path, variant: str, thresholds: list[float]) -> dict | None:
    """Re-evaluate predictions.npz at each detection threshold; returns {label: metrics_dict}, or None if missing."""
    preds_path = results_dir / variant / 'predictions.npz'
    if not preds_path.exists():
        print(f"  WARNING: {preds_path} not found — skipping {variant}")
        return None

    data = np.load(preds_path, allow_pickle=True)
    y_true = data['y_true']
    y_prob = data['y_prob'] if 'y_prob' in data else None
    if y_prob is None:
        raise ValueError(f"{preds_path} has no y_prob — cannot vary alarm threshold.")

    has_timing_meta = all(k in data for k in ('Run_ID', 'start_idx', 'end_idx'))

    # Top-2 margin is independent of the alarm threshold; compute once for context.
    sorted_probs = np.sort(y_prob, axis=1)
    top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    mean_top2_margin = float(top2_margin.mean())
    frac_ambiguous = float((top2_margin < 0.10).mean())

    results = {}
    for t in thresholds:
        alarm_metrics = compute_alarm_metrics(y_prob, y_true, detection_threshold=t)

        timing_metrics = {}
        if has_timing_meta:
            timing = compute_timing_metrics(
                y_prob, data['Run_ID'], data['start_idx'], data['end_idx'],
                detection_threshold=t,
            )
            timing_metrics = {str(k): v for k, v in timing.items()}

        label = format_threshold_label(t)
        results[label] = {
            'alarm_threshold': 1.0 - t,
            'mean_top2_margin': mean_top2_margin,
            'frac_ambiguous': frac_ambiguous,
            'timing_metrics': timing_metrics,
            **alarm_metrics,
        }

    return results


def save_threshold_report(results: dict, out_path: Path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _sheet_alarm_analysis(wb, results)
    _sheet_per_class_fdr(wb, results)
    _sheet_fault_detection_time(wb, results)

    wb.save(out_path)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Alarm-threshold sensitivity analysis (post-hoc, no retraining)'
    )
    parser.add_argument('--model', type=str, default=','.join(DEFAULT_MODELS),
                        help=f'Comma-separated variants whose saved predictions.npz to re-analyze '
                             f'(default: {",".join(DEFAULT_MODELS)}; choices: {",".join(ALL_VARIANTS)})')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--thresholds', type=str,
                        default=','.join(str(t) for t in DEFAULT_THRESHOLDS),
                        help='Comma-separated P(NOC) detection thresholds, e.g. 0.05,0.10,0.15')
    args = parser.parse_args()

    thresholds = sorted(float(t) for t in args.thresholds.split(','))
    models = [m.strip() for m in args.model.split(',')]
    for m in models:
        if m not in ALL_VARIANTS:
            parser.error(f"invalid model '{m}' (choices: {','.join(ALL_VARIANTS)})")

    config = load_config(args.config)
    results_dir = Path(config.results_dir)

    for model in models:
        print("=" * 90)
        print(f"  Alarm-Threshold Sensitivity — {model}")
        print("=" * 90)
        print(f"  Predictions: {results_dir / model / 'predictions.npz'}")
        print(f"  Thresholds:  {[format_threshold_label(t) for t in thresholds]}")
        print("=" * 90)

        results = run_threshold_sensitivity(results_dir, model, thresholds)
        if results is None:
            continue

        out_path = results_dir / model / 'threshold_sensitivity.xlsx'
        save_threshold_report(results, out_path)


if __name__ == '__main__':
    main()
