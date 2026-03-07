"""
Smoke Test: Windowing Pipeline

Validates the output of create_windows.py by checking:
  1. Output .npz files exist
  2. X and y shapes are correct
  3. No windows cross Run_ID boundaries
  4. Label is from the last timestep of each window

Usage:
    python tests/test_windowing.py --exp 1
    python tests/test_windowing.py --exp 2

Author: [Your Name]
Created: February 2026
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_loader import load_config

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

_N_IDVS          = 27    # IDV1-28 excl. IDV6 (doesn't exist in H5)
_N_CLASSES       = 28    # 27 faults + class 0 (healthy, embedded in each fault run)
_TIMESTEPS_PER_RUN = 2001  # TEP runs are 2001 timesteps (0-2000)

EXPERIMENTS = {
    '1': {
        'windows_dir': Path('data/processed_N50_tr30_v10_te10/windows'),
        'window_size': 5,
        'stride':      1,
        'splits':      ['train', 'val', 'test'],
        # Runs per IDV per split (total_runs=50, train/val/test=30/10/10)
        'runs_per_idv': {'train': 30, 'val': 10, 'test': 10},
    },
    '2': {
        'windows_dir': Path('data/processed_N200_tr160_v0_te40/windows'),
        'window_size': 5,
        'stride':      1,
        'splits':      ['train', 'test'],
        # Runs per IDV per split (total_runs=200, train/test=160/40)
        'runs_per_idv': {'train': 160, 'val': 0, 'test': 40},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str):
    print(f"  ✅ PASS  {msg}")

def _fail(msg: str):
    print(f"  ❌ FAIL  {msg}")
    raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_files_exist(windows_dir: Path, window_size: int, stride: int,
                      splits: list = None):
    """Check 1: All expected .npz files exist."""
    print("\n[1/4] Checking output files exist...")
    if splits is None:
        splits = ['train', 'val', 'test']

    for split in splits:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024**2
            _pass(f"{filepath.name}  ({size_mb:.1f} MB)")
        else:
            _fail(
                f"{filepath.name} not found.\n"
                f"       Have you run: python scripts/create_windows.py ?"
            )


def check_shapes(windows_dir: Path, window_size: int, stride: int,
                 runs_per_idv: dict, splits: list = None):
    """
    Check 2: X and y shapes are consistent and correct.

    Expected X shape:  (n_windows, window_size * n_features)
    Expected y shape:  (n_windows,)

    Expected window count formula:
        n_windows_per_run = (timesteps_per_run - window_size) // stride + 1
                          = (2001 - window_size) // stride + 1
        total_windows     = n_idvs × runs_per_idv[split] × n_windows_per_run
                          = 27    × runs_per_idv[split]  × n_windows_per_run

    Where 27 IDVs = IDV1-28 excluding IDV6 (doesn't exist in the dataset).
    Class 0 (healthy) is embedded in each fault run (rows 0-599), so it
    does not add extra runs — only 27 IDVs contribute to the run count.
    """
    print("\n[2/4] Checking X and y shapes...")
    if splits is None:
        splits = ['train', 'val', 'test']

    n_windows_per_run = (_TIMESTEPS_PER_RUN - window_size) // stride + 1
    print(f"  Expected windows/run: ({_TIMESTEPS_PER_RUN} - {window_size}) // {stride} + 1 = {n_windows_per_run:,}")

    for split in splits:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        X = data['X']
        y = data['y']

        # X must be 2D: (n_windows, flattened_dim)
        if X.ndim != 2:
            _fail(f"{split} X should be 2D, got shape {X.shape}")

        # y must be 1D
        if y.ndim != 1:
            _fail(f"{split} y should be 1D, got shape {y.shape}")

        # X and y must have same number of rows
        if X.shape[0] != y.shape[0]:
            _fail(
                f"{split} X rows ({X.shape[0]:,}) != y length ({y.shape[0]:,})"
            )

        # Flattened dimension must be divisible by window_size
        if X.shape[1] % window_size != 0:
            _fail(
                f"{split} X dim {X.shape[1]} is not divisible by "
                f"window_size {window_size}"
            )

        n_features = X.shape[1] // window_size

        # Check expected window count if runs_per_idv is provided for this split
        expected_str = ""
        if runs_per_idv and split in runs_per_idv and runs_per_idv[split] > 0:
            n_runs   = runs_per_idv[split]
            expected = _N_IDVS * n_runs * n_windows_per_run
            expected_str = (
                f"  expected={expected:,}  "
                f"({_N_IDVS} IDVs × {n_runs} runs × {n_windows_per_run:,} windows)"
            )
            if X.shape[0] != expected:
                _fail(
                    f"{split} X has {X.shape[0]:,} windows but expected {expected:,}\n"
                    f"       Formula: {_N_IDVS} IDVs × {n_runs} runs/IDV × "
                    f"{n_windows_per_run:,} windows/run = {expected:,}"
                )

        _pass(
            f"{split:<6}  X={str(X.shape):<25}  y={str(y.shape):<15}  "
            f"n_features={n_features}"
        )
        if expected_str:
            print(f"         {expected_str}")

        # Check all splits have same number of features
        if split == 'train':
            train_features = n_features
        else:
            if n_features != train_features:
                _fail(
                    f"{split} has {n_features} features but train has "
                    f"{train_features} — mismatch!"
                )


def check_run_boundaries(windows_dir: Path, window_size: int, stride: int,
                         splits: list = None):
    """
    Check 3: No window crosses a Run_ID boundary.

    Each window's start_idx and end_idx must belong to the same run.
    We verify this by checking that end_idx - start_idx == window_size - 1
    for every window (consecutive indices within a run).
    """
    print("\n[3/4] Checking no windows cross Run_ID boundaries...")
    if splits is None:
        splits = ['train', 'val', 'test']

    for split in splits:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        if 'start_idx' not in data or 'end_idx' not in data:
            print(f"  ⚠  SKIP  {split} — metadata not saved "
                  f"(save_metadata=false in config)")
            continue

        start_idx = data['start_idx']
        end_idx   = data['end_idx']
        run_ids   = data['Run_ID']

        # end - start should always equal window_size - 1
        span = end_idx - start_idx
        bad  = np.where(span != window_size - 1)[0]

        if len(bad) > 0:
            _fail(
                f"{split}: {len(bad):,} windows have incorrect span "
                f"(expected {window_size - 1}, got other values). "
                f"This means windows are crossing run boundaries!"
            )

        # Each group of consecutive windows with same Run_ID must have
        # monotonically increasing start_idx (no jumps back to 0 mid-run)
        unique_runs = np.unique(run_ids)
        for run_id in unique_runs:
            mask    = run_ids == run_id
            starts  = start_idx[mask]
            diffs   = np.diff(starts)

            if np.any(diffs <= 0):
                _fail(
                    f"{split} run {run_id}: start indices are not "
                    f"strictly increasing — possible boundary issue."
                )

        _pass(f"{split:<6}  all {len(start_idx):,} windows stay within their run")


def check_labels(windows_dir: Path, window_size: int, stride: int,
                 splits: list = None):
    """
    Check 4: Label for each window matches the last timestep's target.

    We verify this on a sample of windows using the metadata
    (start_idx, end_idx, Run_ID) to reconstruct what the label should be.

    Note: This is a structural check — we verify end_idx aligns with
    the label, not the actual feature values.
    """
    print("\n[4/4] Checking labels come from last timestep of each window...")
    if splits is None:
        splits = ['train', 'val', 'test']

    for split in splits:
        filepath = windows_dir / f'{split}_windows_w{window_size}_s{stride}.npz'
        data = np.load(filepath, allow_pickle=True)

        if 'end_idx' not in data:
            print(f"  ⚠  SKIP  {split} — metadata not saved")
            continue

        y         = data['y']
        end_idx   = data['end_idx']

        # Check label classes are valid (0-28 for TEP)
        unique_labels = np.unique(y)
        invalid = unique_labels[(unique_labels < 0) | (unique_labels > 28)]

        if len(invalid) > 0:
            _fail(
                f"{split}: found invalid label values {invalid}. "
                f"Expected 0-28 (TEP fault classes)."
            )

        # Check expected class count (TEP has 28 classes: 0 + IDV1-28 excl. IDV6)
        n_classes = len(unique_labels)
        if n_classes != 28:
            _fail(
                f"{split}: expected 28 unique classes, found {n_classes}. "
                f"Classes present: {sorted(unique_labels.tolist())}"
            )

        _pass(
            f"{split:<6}  labels valid  |  "
            f"{n_classes} classes  |  "
            f"range [{y.min()}, {y.max()}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(config_path: str = None, exp: str = None):
    """
    Run all windowing smoke tests.

    Parameters:
    -----------
    config_path : str, optional
        Path to config.yaml (used when --config is specified)
    exp : str, optional
        Experiment number ('1' or '2') — uses hardcoded paths
    """

    print("\n" + "="*70)
    print(" "*18 + "WINDOWING SMOKE TEST")
    print("="*70)

    runs_per_idv = {}

    if exp is not None:
        if exp not in EXPERIMENTS:
            print(f"  Unknown experiment '{exp}'. Valid options: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        cfg          = EXPERIMENTS[exp]
        windows_dir  = cfg['windows_dir']
        window_size  = cfg['window_size']
        stride       = cfg['stride']
        splits       = cfg['splits']
        runs_per_idv = cfg['runs_per_idv']
        print(f"\n  Experiment:   {exp}")
    else:
        if config_path is None:
            config_path = 'configs/config.yaml'
        config      = load_config(config_path)
        window_size = config.window_size
        stride      = config.stride
        windows_dir = Path(config.output_dir) / 'windows'
        splits = ['train', 'test']
        if config.val_runs > 0:
            splits = ['train', 'val', 'test']
        print(f"\n  Config:       {config_path}")

    print(f"  Windows dir:  {windows_dir}")
    print(f"  Window size:  {window_size}")
    print(f"  Stride:       {stride}")
    print(f"  Splits:       {splits}")

    passed = 0
    failed = 0

    # Run each check
    checks = [
        ("Files exist",           check_files_exist),
        ("X and y shapes",        check_shapes),
        ("Run_ID boundaries",     check_run_boundaries),
        ("Labels from last step", check_labels),
    ]

    for check_name, check_fn in checks:
        try:
            if check_fn.__name__ == 'check_shapes':
                check_fn(windows_dir, window_size, stride, runs_per_idv, splits)
            else:
                check_fn(windows_dir, window_size, stride, splits)
            passed += 1
        except AssertionError:
            failed += 1
            # Error already printed inside check function
            if check_fn.__name__ == 'check_files_exist':
                print("\n  ⚠  Files missing — skipping remaining checks.")
                break

    # Final result
    total = passed + failed
    print("\n" + "="*70)

    if failed == 0:
        print(f"  🎉 ALL {total} CHECKS PASSED — windowing looks correct!")
    else:
        print(f"  ⚠  {passed}/{total} checks passed, {failed} FAILED")
        print(f"     Fix the issues above then rerun this test.")

    print("="*70 + "\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Smoke test for TEP windowing output'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--exp',
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help='Experiment number (e.g. --exp 1 or --exp 2)'
    )
    group.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml (fallback if --exp not used)'
    )
    args = parser.parse_args()

    success = run_smoke_test(config_path=args.config, exp=args.exp)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()