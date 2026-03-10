#!/usr/bin/env python3
# Create sliding windows from processed TEP data (train/val/test_final.pkl).
# Usage: python scripts/create_windows.py [--config configs/config.yaml]

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_loader import load_config
from src.windowing.tep_windowing import create_all_windows, save_windows


def run_windowing_pipeline(config_path: str = 'configs/config.yaml'):
    """Load processed dataframes and create sliding windows; returns windowed data dict."""

    config = load_config(config_path)
    config.display()

    processed_dir = Path(config.output_dir)
    windows_dir   = processed_dir / 'windows'

    train_pkl = processed_dir / 'train_final.pkl'
    val_pkl   = processed_dir / 'val_final.pkl'
    test_pkl  = processed_dir / 'test_final.pkl'

    window_size   = config.window_size
    stride        = config.stride
    save_metadata = config.save_metadata

    print("\n" + "="*70)
    print(" "*20 + "TEP WINDOWING PIPELINE")
    print("="*70)
    print(f"\n  Input:   {processed_dir}")
    print(f"  Output:  {windows_dir}")
    print(f"  Window:  {window_size}  Stride: {stride}  Metadata: {save_metadata}")
    print("="*70 + "\n")

    # Load processed dataframes
    print("Step 1/3: Loading processed data")
    print("-" * 40)

    for pkl_path in [train_pkl, test_pkl]:
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"File not found: {pkl_path}\n"
                f"Have you run scripts/run_pipeline.py first?"
            )

    print("  Loading train_final.pkl...")
    train_df = pd.read_pickle(train_pkl)
    print(f"  Train: {train_df.shape}")

    if val_pkl.exists():
        print("  Loading val_final.pkl...")
        val_df = pd.read_pickle(val_pkl)
        print(f"  Val:   {val_df.shape}")
    else:
        print("  val_final.pkl not found — skipping validation split")
        val_df = None

    print("  Loading test_final.pkl...")
    test_df = pd.read_pickle(test_pkl)
    print(f"  Test:  {test_df.shape}")

    # Create windows
    print("\nStep 2/3: Creating sliding windows")
    print("-" * 40)

    windows_data = create_all_windows(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        window_size=window_size,
        stride=stride
    )

    # Save
    print("\nStep 3/3: Saving windows")
    print("-" * 40)

    saved_paths = save_windows(
        windows_data=windows_data,
        output_dir=windows_dir,
        window_size=window_size,
        stride=stride,
        save_metadata=save_metadata
    )

    # Summary
    X_train = windows_data['train'][0]
    print("\n" + "="*70)
    print(f"  Window shape: n_windows x {window_size} x {X_train.shape[1] // window_size} (flattened)")
    print(f"  {'Split':<10} {'Windows':>12}   {'File'}")
    print(f"  {'-'*50}")
    for split, path in saved_paths.items():
        n = len(windows_data[split][0])
        print(f"  {split:<10} {n:>12,}   {path.name}")
    print(f"\n  Saved to: {windows_dir}")
    print("="*70 + "\n")

    return windows_data


def main():
    parser = argparse.ArgumentParser(description='Create sliding windows from processed TEP data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file (default: configs/config.yaml)')
    args = parser.parse_args()
    run_windowing_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
