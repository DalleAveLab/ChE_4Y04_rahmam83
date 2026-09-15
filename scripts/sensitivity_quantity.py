#!/usr/bin/env python3
# Quantity-of-data sensitivity analysis: aggregates already-computed eval_metrics.json
# from per-train-size experiment configs (fixed N=200, varying train/test split) into
# one comparison workbook.
# Usage: python scripts/sensitivity_quantity.py --model wavelet_kan
#        python scripts/sensitivity_quantity.py --model wavelet_kan --configs configs/sensitivity_quantity_tr40.yaml,...

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
    'configs/config.yaml',               # tr160 baseline (Experiment 2)
    'configs/sensitivity_quantity_tr80.yaml',
    'configs/sensitivity_quantity_tr40.yaml',
    'configs/sensitivity_quantity_tr20.yaml',
    'configs/sensitivity_quantity_tr10.yaml',
    'configs/sensitivity_quantity_tr5.yaml',
]


def load_quantity_results(config_paths: list[str], model: str) -> dict:
    """Load eval_metrics.json for `model` from each config's results_dir; returns {'tr{train}': metrics_dict}."""
    results = {}
    for config_path in config_paths:
        config = load_config(config_path)
        label = f"tr{config.train_runs}"
        metrics_path = Path(config.results_dir) / model / 'eval_metrics.json'
        if not metrics_path.exists():
            print(f"  WARNING: {metrics_path} not found — skipping {label}")
            continue
        with open(metrics_path, 'r') as f:
            results[label] = json.load(f)
        print(f"  Loaded {label}: {metrics_path}")
    return results


def save_quantity_report(results: dict, out_path: Path):
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
        description='Quantity-of-data sensitivity analysis (aggregates per-run-count eval_metrics.json)'
    )
    parser.add_argument('--model', type=str, default='wavelet_kan',
                        choices=ALL_VARIANTS,
                        help='Variant to compare across run counts')
    parser.add_argument('--configs', type=str,
                        default=','.join(DEFAULT_CONFIGS),
                        help='Comma-separated config paths, one per train size (descending order)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output xlsx path (default: quantity_sensitivity_<model>.xlsx)')
    args = parser.parse_args()

    config_paths = [c.strip() for c in args.configs.split(',')]
    out_path = Path(args.out) if args.out else Path(f'quantity_sensitivity_{args.model}.xlsx')

    print("=" * 90)
    print(f"  Quantity-of-Data Sensitivity — {args.model}")
    print("=" * 90)

    results = load_quantity_results(config_paths, args.model)
    if not results:
        print("  No results found — nothing to report.")
        return

    save_quantity_report(results, out_path)


if __name__ == '__main__':
    main()
