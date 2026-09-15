"""
Tennessee Eastman Process (TEP) Sliding Window Module

Creates sliding windows from processed TEP time-series data.
Windows are created per run (grouped by Run_ID) to preserve
temporal integrity and avoid cross-run contamination.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional


class TEPWindowGenerator:
    """
    Generates sliding windows from TEP time-series DataFrames.

    Windows are always created within individual runs (grouped by Run_ID)
    so that no window spans across two different simulation runs.

    Each window's label is taken from the LAST timestep in the window.
    """

    def __init__(self, window_size: int = 5, stride: int = 1):
        """
        Initialize the window generator.

        Parameters:
        -----------
        window_size : int
            Number of consecutive timesteps per window (default: 5)
        stride : int
            Number of timesteps to advance between windows (default: 1)
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = None
        self.num_features = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """
        Identify feature columns (exclude Run_ID and Target).

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame

        Returns:
        --------
        list
            List of feature column names
        """
        return [col for col in df.columns if col not in ['Run_ID', 'Target']]

    def create_windows(self, df: pd.DataFrame,
                       split_name: str = "Data") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create sliding windows from a DataFrame.

        Windows are created within each Run_ID group to preserve
        temporal integrity. The label for each window is taken from
        the last timestep in the window.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: features, 'Run_ID', 'Target'
        split_name : str
            Name of the split (for progress bar display)

        Returns:
        --------
        X : np.ndarray
            Shape: (n_windows, window_size, n_features)
        y : np.ndarray
            Shape: (n_windows,) - label from last timestep of each window
        window_info : pd.DataFrame
            Metadata: Run_ID, start_idx, end_idx for each window
        """

        # Identify feature columns
        self.feature_cols = self._get_feature_cols(df)
        self.num_features = len(self.feature_cols)

        print(f"\nCreating sliding windows for {split_name}:")
        print(f"  Window size:  {self.window_size} timesteps")
        print(f"  Stride:       {self.stride} timestep(s)")
        print(f"  Features:     {self.num_features}")
        print(f"  Unique runs:  {df['Run_ID'].nunique()}")

        X_list = []
        y_list = []
        metadata_list = []

        unique_runs = df['Run_ID'].unique()

        for run_id in tqdm(unique_runs, desc=f"  Windowing {split_name}"):
            # Extract this run's data
            run_data = df[df['Run_ID'] == run_id].reset_index(drop=True)

            features = run_data[self.feature_cols].values  # (2001, n_features)
            targets = run_data['Target'].values             # (2001,)

            n_samples = len(run_data)
            n_windows = (n_samples - self.window_size) // self.stride + 1

            if n_windows <= 0:
                continue

            # Vectorized window creation
            start_indices = np.arange(0, n_windows * self.stride, self.stride)
            window_indices = start_indices[:, None] + np.arange(self.window_size)

            # Shape: (n_windows, window_size, n_features) → flatten for KAN
            windows = features[window_indices]
            windows_flat = windows.reshape(n_windows, -1)  # (n_windows, window_size * n_features)

            # Label = last timestep of each window
            labels = targets[window_indices[:, -1]]

            X_list.append(windows_flat)
            y_list.append(labels)

            metadata_list.append(pd.DataFrame({
                'Run_ID': run_id,
                'start_idx': start_indices,
                'end_idx': start_indices + self.window_size - 1
            }))

        # Concatenate all runs
        X = np.vstack(X_list)                              # (total_windows, window_size * n_features)
        y = np.concatenate(y_list)                         # (total_windows,)
        window_info = pd.concat(metadata_list, ignore_index=True)

        print(f"  ✓ X shape: {X.shape}  →  (n_windows, window_size × n_features)")
        print(f"  ✓ y shape: {y.shape}")
        print(f"  ✓ Unique classes: {len(np.unique(y))}")
        print(f"  ✓ Flattened dim: {self.window_size} × {self.num_features} = {X.shape[1]}")

        return X, y, window_info


def create_all_windows(train_df: pd.DataFrame,
                       val_df: pd.DataFrame = None,
                       test_df: pd.DataFrame = None,
                       window_size: int = 5,
                       stride: int = 1) -> dict:
    """
    Create sliding windows for all three splits.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame (from train_final.pkl)
    val_df : pd.DataFrame
        Validation DataFrame (from val_final.pkl)
    test_df : pd.DataFrame
        Test DataFrame (from test_final.pkl)
    window_size : int
        Number of timesteps per window
    stride : int
        Step size between windows

    Returns:
    --------
    dict with keys 'train', 'val', 'test'
        Each value is a tuple: (X, y, window_info)
        X shape: (n_windows, window_size * n_features)  ← flattened for KAN
        y shape: (n_windows,)
    """

    if test_df is None:
        raise ValueError("test_df is required")

    print("\n" + "="*70)
    print("SLIDING WINDOW CREATION")
    print("="*70)

    generator = TEPWindowGenerator(window_size=window_size, stride=stride)

    print("\n[1/3] Training data...")
    X_train, y_train, train_info = generator.create_windows(train_df, "Train")

    if val_df is not None:
        print("\n[2/3] Validation data...")
        X_val, y_val, val_info = generator.create_windows(val_df, "Validation")
    else:
        print("\n[2/3] Validation data — skipped (val_runs=0)")
        X_val, y_val, val_info = None, None, None

    print("\n[3/3] Test data...")
    X_test, y_test, test_info = generator.create_windows(test_df, "Test")

    # Summary
    total = len(X_train) + len(X_test) + (len(X_val) if X_val is not None else 0)
    print("\n" + "="*70)
    print("WINDOWING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n  Window size:  {window_size} timesteps")
    print(f"  Stride:       {stride} timestep(s)")
    print(f"  Input dim:    {window_size} × {X_train.shape[1] // window_size} = {X_train.shape[1]}  (window_size × n_features, flattened for KAN)")
    print(f"\n  {'Split':<12} {'Windows':>12} {'Shape':>30}")
    print(f"  {'-'*55}")
    print(f"  {'Train':<12} {len(X_train):>12,} {str(X_train.shape):>30}")
    if X_val is not None:
        print(f"  {'Val':<12} {len(X_val):>12,} {str(X_val.shape):>30}")
    print(f"  {'Test':<12} {len(X_test):>12,} {str(X_test.shape):>30}")
    print(f"  {'Total':<12} {total:>12,}")
    print("="*70 + "\n")

    result = {
        'train': (X_train, y_train, train_info),
        'test':  (X_test,  y_test,  test_info),
    }
    if X_val is not None:
        result['val'] = (X_val, y_val, val_info)
    return result


def save_windows(windows_data: dict,
                 output_dir: str,
                 window_size: int,
                 stride: int,
                 save_metadata: bool = True):
    """
    Save windowed arrays to compressed .npz files.

    Output filenames include window/stride config for easy identification:
        train_windows_w5_s1.npz
        val_windows_w5_s1.npz
        test_windows_w5_s1.npz

    Parameters:
    -----------
    windows_data : dict
        Output from create_all_windows()
    output_dir : str
        Directory to save files (e.g., 'data/processed/windows/')
    window_size : int
        Window size (used in filename)
    stride : int
        Stride (used in filename)
    save_metadata : bool
        Whether to include Run_ID/start_idx/end_idx in the .npz file
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SAVING WINDOWED DATA")
    print("="*70)
    print(f"  Output directory: {output_path}")
    print(f"  Window size: {window_size}, Stride: {stride}\n")

    splits = {
        'train': '1/3',
        'val':   '2/3',
        'test':  '3/3'
    }

    saved_paths = {}

    for split_name, step in splits.items():
        if split_name not in windows_data:
            print(f"  [{step}] Skipping {split_name} (not available)")
            continue
        X, y, info = windows_data[split_name]

        filepath = output_path / f'{split_name}_windows_w{window_size}_s{stride}.npz'

        print(f"  [{step}] Saving {split_name}...")

        if save_metadata:
            np.savez_compressed(
                filepath,
                X=X,
                y=y,
                Run_ID=info['Run_ID'].values,
                start_idx=info['start_idx'].values,
                end_idx=info['end_idx'].values
            )
        else:
            np.savez_compressed(filepath, X=X, y=y)

        size_mb = filepath.stat().st_size / 1024**2
        print(f"       ✓ {filepath.name}")
        print(f"         X shape: {X.shape}, y shape: {y.shape}")
        print(f"         Size: {size_mb:.2f} MB")

        saved_paths[split_name] = filepath

    total_mb = sum(p.stat().st_size for p in saved_paths.values()) / 1024**2

    print(f"\n  ✓ All files saved  |  Total size: {total_mb:.2f} MB")
    print("="*70 + "\n")

    return saved_paths


def load_windows(windows_dir: str, window_size: int, stride: int) -> dict:
    """
    Load previously saved window .npz files.

    Parameters:
    -----------
    windows_dir : str
        Directory containing the .npz files
    window_size : int
        Window size used when creating (for filename matching)
    stride : int
        Stride used when creating (for filename matching)

    Returns:
    --------
    dict with keys 'train', 'val', 'test'
        Each value is a dict: {'X': ..., 'y': ..., 'Run_ID': ..., ...}
    """

    windows_path = Path(windows_dir)
    result = {}

    for split in ['train', 'val', 'test']:
        filepath = windows_path / f'{split}_windows_w{window_size}_s{stride}.npz'

        if not filepath.exists():
            raise FileNotFoundError(f"Window file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        result[split] = dict(data)
        print(f"  ✓ Loaded {split}: X={data['X'].shape}, y={data['y'].shape}")

    return result