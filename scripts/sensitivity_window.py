#!/usr/bin/env python3
# Window-size sensitivity analysis: aggregates already-computed eval_metrics.json
# from a set of per-window-size experiment configs into one comparison workbook.
# Usage: python scripts/sensitivity_window.py --model wavelet_kan
#        python scripts/sensitivity_window.py --model wavelet_kan --configs configs/sensitivity_window_1.yaml,configs/config.yaml,...

import sys
import json
import argparse
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openpyxl

from configs.config_loader import load_config
from scripts.evaluate import (
    _sheet_model_comparison, _sheet_per_class_f1, _sheet_alarm_analysis,
    _sheet_per_class_fdr, _sheet_fault_detection_time, _sheet_fault_diagnosis_time,
    ALL_VARIANTS,
)

DEFAULT_CONFIGS = [
    'configs/sensitivity_window_1.yaml',
    'configs/sensitivity_window_3.yaml',
    'configs/config.yaml',              # window_size=5 baseline
    'configs/sensitivity_window_7.yaml',
    'configs/sensitivity_window_9.yaml',
]


def load_window_results(config_paths: list[str], model: str) -> dict:
    """Load eval_metrics.json for `model` from each config's results_dir; returns {'w{N}': metrics_dict}."""
    results = {}
    for config_path in config_paths:
        config = load_config(config_path)
        label = f"w{config.window_size}"
        metrics_path = Path(config.results_dir) / model / 'eval_metrics.json'
        if not metrics_path.exists():
            print(f"  WARNING: {metrics_path} not found — skipping {label}")
            continue
        with open(metrics_path, 'r') as f:
            results[label] = json.load(f)
        print(f"  Loaded {label}: {metrics_path}")
    return results


def save_window_report(results: dict, out_path: Path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _sheet_model_comparison(wb, results)
    _sheet_per_class_f1(wb, results)
    _sheet_alarm_analysis(wb, results)
    _sheet_per_class_fdr(wb, results)
    _sheet_fault_detection_time(wb, results)
    _sheet_fault_diagnosis_time(wb, results)

    wb.save(out_path)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Window-size sensitivity analysis (aggregates per-window-size eval_metrics.json)'
    )
    parser.add_argument('--model', type=str, default='wavelet_kan',
                        choices=ALL_VARIANTS,
                        help='Variant to compare across window sizes')
    parser.add_argument('--configs', type=str,
                        default=','.join(DEFAULT_CONFIGS),
                        help='Comma-separated config paths, one per window size')
    parser.add_argument('--out', type=str, default=None,
                        help='Output xlsx path (default: window_sensitivity_<model>.xlsx)')
    args = parser.parse_args()

    config_paths = [c.strip() for c in args.configs.split(',')]
    out_path = Path(args.out) if args.out else Path(f'window_sensitivity_{args.model}.xlsx')

    print("=" * 90)
    print(f"  Window-Size Sensitivity — {args.model}")
    print("=" * 90)

    results = load_window_results(config_paths, args.model)
    if not results:
        print("  No results found — nothing to report.")
        return

    save_window_report(results, out_path)


if __name__ == '__main__':
    main()
