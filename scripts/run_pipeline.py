#!/usr/bin/env python3
# TEP preprocessing pipeline: process → feature engineering → scaling → windowing.
# Usage: python scripts/run_pipeline.py [--config configs/config.yaml]

import sys
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing.tep_data_processor import TEPDataProcessor
from src.preprocessing.tep_feature_engineering import apply_feature_engineering
from src.preprocessing.tep_data_scaler import scale_dataframes
from configs.config_loader import load_config
import pandas as pd
import argparse


def run_complete_pipeline(config_path: str = 'configs/config.yaml'):
    """Run the full TEP preprocessing pipeline and return processed dataframes + metadata."""

    config = load_config(config_path)
    config.display()

    source_file = config.raw_source
    output_dir = config.output_dir
    random_seed = config.random_seed

    random.seed(random_seed)
    np.random.seed(random_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(" "*20 + "TEP DATA PROCESSING PIPELINE")
    print("="*80 + "\n")

    # Step 1: extract + split + label
    print("Step 1/4: Data Processing")
    print("-" * 40)

    processor = TEPDataProcessor(
        source_file, random_seed=random_seed,
        total_runs=config.total_runs, train_runs=config.train_runs,
        val_runs=config.val_runs,     test_runs=config.test_runs,
    )

    print("Extracting and splitting data...")
    train_file, val_file, test_file = processor.extract_and_split_data(output_dir)

    print("Loading and labeling data...")
    train_df_raw = processor.load_and_label_data(train_file, split_name='train')
    val_df_raw = (processor.load_and_label_data(val_file, split_name='validation')
                  if val_file is not None else None)
    test_df_raw = processor.load_and_label_data(test_file, split_name='test')

    print(f"  Train: {train_df_raw.shape}")
    if val_df_raw is not None:
        print(f"  Val:   {val_df_raw.shape}")
    print(f"  Test:  {test_df_raw.shape}")

    # Step 2: drop zero-variance + analyzer features
    print("\nStep 2/4: Feature Engineering")
    print("-" * 40)

    fe_save_path = output_path / 'feature_engineer.pkl'

    train_df_fe, val_df_fe, test_df_fe = apply_feature_engineering(
        train_df=train_df_raw,
        val_df=val_df_raw,
        test_df=test_df_raw,
        drop_analyzers=config.drop_analyzers,
        drop_zero_var=config.drop_zero_variance,
        variance_threshold=config.variance_threshold,
        analyzer_list=config.analyzer_features,
        save_path=fe_save_path
    )

    print(f"  Features: {train_df_raw.shape[1] - 2} -> {train_df_fe.shape[1] - 2} "
          f"({(train_df_raw.shape[1] - 2) - (train_df_fe.shape[1] - 2)} dropped)")

    # Step 3: z-score normalization, fit on train only
    print("\nStep 3/4: Scaling (Z-score)")
    print("-" * 40)

    scaler_save_path = output_path / 'scaler.pkl'

    train_df_scaled, val_df_scaled, test_df_scaled, scaler = scale_dataframes(
        train_df=train_df_fe,
        val_df=val_df_fe,
        test_df=test_df_fe,
        save_scaler_path=scaler_save_path
    )

    feature_cols = [col for col in train_df_scaled.columns
                    if col not in ['Run_ID', 'Target', 'Time']]

    print(f"  Scaled {len(feature_cols)} features (Time, Run_ID, Target preserved)")

    # Step 4: verify
    print("\nStep 4/4: Verification")
    print("-" * 40)

    processor.verify_dataframe(train_df_scaled, 'Train (Final)')
    if val_df_scaled is not None:
        processor.verify_dataframe(val_df_scaled, 'Validation (Final)')
    processor.verify_dataframe(test_df_scaled, 'Test (Final)')

    # Save final dataframes
    if config.save_dataframes:
        print("\nSaving final dataframes...")
        train_df_scaled.to_pickle(output_path / 'train_final.pkl')
        test_df_scaled.to_pickle(output_path / 'test_final.pkl')
        print(f"  {output_path / 'train_final.pkl'}")

        if val_df_scaled is not None:
            val_df_scaled.to_pickle(output_path / 'val_final.pkl')
            print(f"  {output_path / 'val_final.pkl'}")

        print(f"  {output_path / 'test_final.pkl'}")

        if config.save_csv_sample:
            train_df_scaled.head(config.csv_sample_rows).to_csv(
                output_path / 'train_final_sample.csv', index=False
            )
            print(f"  {output_path / 'train_final_sample.csv'} (first {config.csv_sample_rows:,} rows)")

    # Summary
    print("\n" + "="*80)
    print("Pipeline complete")
    print(f"  Features: {train_df_raw.shape[1] - 2} -> {len(feature_cols)} (after eng + scaling)")
    print(f"  Train: {train_df_scaled.shape[0]:,} x {train_df_scaled.shape[1]}")
    if val_df_scaled is not None:
        print(f"  Val:   {val_df_scaled.shape[0]:,} x {val_df_scaled.shape[1]}")
    print(f"  Test:  {test_df_scaled.shape[0]:,} x {test_df_scaled.shape[1]}")
    print(f"  Output: {output_path}/")
    print("="*80 + "\n")

    return {
        'train_df': train_df_scaled,
        'val_df': val_df_scaled,
        'test_df': test_df_scaled,
        'scaler': scaler,
        'feature_engineer_path': fe_save_path,
        'scaler_path': scaler_save_path,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols
    }


def main():
    parser = argparse.ArgumentParser(description='TEP preprocessing pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file (default: configs/config.yaml)')
    args = parser.parse_args()
    return run_complete_pipeline(config_path=args.config)


if __name__ == "__main__":
    results = main()
