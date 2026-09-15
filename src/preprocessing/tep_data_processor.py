"""
Tennessee Eastman Process (TEP) Data Processing Pipeline

This module provides utilities for:
1. Extracting and splitting data from the main HDF5 file
2. Loading and labeling the data with proper column names and target labels
3. Combining processdata and additional_meas datasets

"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
from typing import Tuple, Optional, List


class TEPDataProcessor:
    """
    Main class for processing Tennessee Eastman Process data.
    
    Handles extraction, splitting, loading, and labeling of TEP data from HDF5 files.
    """
    
    def __init__(self, source_file: str, random_seed: int = 42,
                 total_runs: int = 50, train_runs: int = 30,
                 val_runs: int = 10, test_runs: int = 10):
        """
        Initialize the TEP Data Processor.

        Parameters:
        -----------
        source_file : str
            Path to the source HDF5 file (e.g., 'TEP_Mode1.h5')
        random_seed : int
            Random seed for reproducibility
        total_runs : int
            Number of runs to randomly select per IDV (default: 50)
        train_runs : int
            Number of runs allocated to training (default: 30)
        val_runs : int
            Number of runs allocated to validation; set 0 to skip (default: 10)
        test_runs : int
            Number of runs allocated to testing (default: 10)
        """
        self.source_file = Path(source_file)
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Split configuration (read from args, not hardcoded)
        self.TOTAL_RUNS = total_runs
        self.TRAIN_RUNS = train_runs
        self.VAL_RUNS = val_runs
        self.TEST_RUNS = test_runs
        self.FAULT_START_INDEX = 600  # Fault is inserted at index 600
        
        # IDV range (1-28, excluding 6 which doesn't exist)
        self.IDV_RANGE = range(1, 29)
        
    def extract_and_split_data(self, output_dir: str = '.') -> Tuple[Path, Path, Path]:
        """
        Extract 50 random runs from each IDV and split into train/val/test sets.
        Combines processdata and additional_meas into a single dataset.
        
        Parameters:
        -----------
        output_dir : str
            Directory where output files will be saved
        
        Returns:
        --------
        Tuple[Path, Path, Path]
            Paths to (train_file, val_file, test_file)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define output file paths (filenames reflect actual run counts)
        train_file = output_path / f'train_{self.TRAIN_RUNS}runs.h5'
        val_file   = output_path / f'val_{self.VAL_RUNS}runs.h5' if self.VAL_RUNS > 0 else None
        test_file  = output_path / f'test_{self.TEST_RUNS}runs.h5'
        
        print(f"\n{'='*70}")
        print(f"EXTRACTING AND SPLITTING TEP DATA")
        print(f"{'='*70}")
        print(f"Source: {self.source_file}")
        print(f"Random seed: {self.random_seed}")
        print(f"Split: {self.TRAIN_RUNS} train / {self.VAL_RUNS} val / {self.TEST_RUNS} test")
        print(f"{'='*70}\n")
        
        with h5py.File(self.source_file, 'r') as src:
            
            # Load and save labels to all output files
            print("Loading labels...")
            processdata_labels = src['Processdata_Labels'][:].astype(str)
            additional_meas_labels = src['Additional_Meas_Labels'][:].astype(str)
            
            # Combine labels
            combined_labels = np.concatenate([processdata_labels, additional_meas_labels])
            print(f"✓ Combined {len(processdata_labels)} processdata + {len(additional_meas_labels)} additional_meas labels")
            print(f"  Total features: {len(combined_labels)}")
            
            # Save labels to output files (skip val_file when val_runs=0)
            output_files = [f for f in [train_file, val_file, test_file] if f is not None]
            for out_file in output_files:
                with h5py.File(out_file, 'w') as dst:
                    dst.create_dataset('Feature_Labels', data=combined_labels.astype('S'))
            print(f"✓ Saved labels to output files\n")
            
            # Process each IDV
            for idv_num in tqdm(self.IDV_RANGE, desc="Processing IDVs"):
                # Construct path to the 100% magnitude group
                idv_path = f"Mode1/SingleFault/SimulationCompleted/IDV{idv_num}/Mode1_IDVInfo_{idv_num}_100"
                
                # Check if path exists (IDV6 won't exist)
                if idv_path not in src:
                    continue
                
                idv_group = src[idv_path]
                
                # Get all run names
                all_runs = [key for key in idv_group.keys() if key.startswith('Run')]
                
                if len(all_runs) < self.TOTAL_RUNS:
                    print(f"  Warning: IDV{idv_num} has only {len(all_runs)} runs (need {self.TOTAL_RUNS})")
                    continue
                
                # Randomly select 50 runs
                selected_runs = np.random.choice(all_runs, size=self.TOTAL_RUNS, replace=False)
                
                # Split into train/val/test
                train_runs = selected_runs[:self.TRAIN_RUNS]
                val_runs   = selected_runs[self.TRAIN_RUNS:self.TRAIN_RUNS + self.VAL_RUNS]
                test_runs  = selected_runs[self.TRAIN_RUNS + self.VAL_RUNS:]

                # Copy runs to respective output files (skip val when val_runs=0)
                splits = [(train_file, train_runs, 'train'),
                          (test_file,  test_runs,  'test')]
                if self.VAL_RUNS > 0:
                    splits.insert(1, (val_file, val_runs, 'val'))
                
                for out_file, runs, split_name in splits:
                    with h5py.File(out_file, 'a') as dst:
                        # Create IDV group if it doesn't exist
                        idv_group_name = f"IDV{idv_num}"
                        if idv_group_name not in dst:
                            dst.create_group(idv_group_name)
                        
                        idv_dst_group = dst[idv_group_name]
                        
                        # Copy each run's combined data
                        for run_name in runs:
                            run_src = idv_group[run_name]
                            
                            # Extract processdata and additional_meas
                            if 'processdata' not in run_src or 'additional_meas' not in run_src:
                                print(f"    Warning: Missing data in {run_name}, skipping...")
                                continue
                            
                            processdata = run_src['processdata'][:]
                            additional_meas = run_src['additional_meas'][:]
                            
                            # Combine the two datasets horizontally
                            combined_data = np.hstack([processdata, additional_meas])
                            
                            # Save to destination
                            idv_dst_group.create_dataset(run_name, data=combined_data)
        
        print(f"\n{'='*70}")
        print(f"✓ DATA EXTRACTION AND SPLITTING COMPLETE!")
        print(f"{'='*70}")
        print(f"Output files created:")
        print(f"  - {train_file}")
        if val_file is not None:
            print(f"  - {val_file}")
        print(f"  - {test_file}")
        print(f"{'='*70}\n")

        return train_file, val_file, test_file
    
    def load_and_label_data(self, file_path: str, split_name: str = None) -> pd.DataFrame:
        """
        Load HDF5 data and create a labeled DataFrame with proper column names and targets.
        
        Parameters:
        -----------
        file_path : str
            Path to the HDF5 file
        split_name : str, optional
            Name of the split for display purposes (e.g., 'train', 'val', 'test')
        
        Returns:
        --------
        pd.DataFrame
            Consolidated DataFrame with all runs, proper column names, Run_ID, and Target
        """
        
        split_label = split_name.upper() if split_name else Path(file_path).stem.upper()
        
        print(f"\n{'='*70}")
        print(f"LOADING AND LABELING: {split_label}")
        print(f"{'='*70}")
        print(f"File: {file_path}\n")
        
        with h5py.File(file_path, 'r') as f:
            
            # Load column names from Feature_Labels
            feature_labels = f['Feature_Labels'][:].astype(str)
            
            # Deduplicate column names (processdata and additional_meas
            # share some names like 'A Feed', 'E Feed', 'A and C Feed')
            cols = [str(c) for c in feature_labels]  # convert all to Python str first
            seen = {}
            for i, col in enumerate(cols):
                if col in seen:
                    seen[col] += 1
                    cols[i] = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
            feature_labels = cols
            
            n_dupes = sum(1 for c in feature_labels if '_' in c and c.rsplit('_', 1)[-1].isdigit())
            print(f"✓ Loaded {len(feature_labels)} feature labels ({n_dupes} duplicates renamed)")
            
            # List to store all DataFrames
            dataframes_list = []
            
            # Count total runs for progress bar
            total_runs = 0
            for idv_num in self.IDV_RANGE:
                idv_key = f'IDV{idv_num}'
                if idv_key in f:
                    total_runs += len([k for k in f[idv_key].keys() if k.startswith('Run')])
            
            print(f"✓ Found {total_runs} total runs to process\n")
            
            # Create progress bar
            with tqdm(total=total_runs, desc="Processing runs") as pbar:
                
                # Iterate through all IDV groups
                for idv_num in self.IDV_RANGE:
                    idv_key = f'IDV{idv_num}'
                    
                    # Check if this IDV exists in the file
                    if idv_key not in f:
                        continue
                    
                    idv_group = f[idv_key]
                    
                    # Get all runs in this IDV group
                    run_keys = [key for key in idv_group.keys() if key.startswith('Run')]
                    
                    if len(run_keys) == 0:
                        continue
                    
                    # Process each run
                    for run_key in run_keys:
                        # Load combined data
                        combined_data = idv_group[run_key][:]
                        
                        # Create DataFrame with proper column names
                        df = pd.DataFrame(combined_data, columns=feature_labels)
                        
                        # Add Run_ID column
                        run_id = f'{idv_key}_{run_key}'
                        df['Run_ID'] = run_id
                        
                        # Add Target column
                        # Rows 0-599: Healthy (0)
                        # Rows 600-end: Faulty (IDV number)
                        n_rows = len(df)
                        target = np.zeros(n_rows, dtype=np.int32)
                        target[self.FAULT_START_INDEX:] = idv_num
                        df['Target'] = target
                        
                        # Append to list
                        dataframes_list.append(df)
                        
                        # Update progress bar
                        pbar.update(1)
        
        # Concatenate all DataFrames
        print(f"\n✓ Concatenating {len(dataframes_list)} DataFrames...")
        final_df = pd.concat(dataframes_list, axis=0, ignore_index=True)
        
        # Free memory
        del dataframes_list
        gc.collect()
        
        print(f"✓ Final DataFrame shape: {final_df.shape}")
        print(f"  - Rows: {final_df.shape[0]:,}")
        print(f"  - Columns: {final_df.shape[1]}")
        print(f"✓ Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return final_df
    
    def verify_dataframe(self, df: pd.DataFrame, split_name: str):
        """
        Verify the DataFrame structure and labeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to verify
        split_name : str
            Name of the split (e.g., 'Train', 'Validation', 'Test')
        """
        
        print(f"\n{'='*70}")
        print(f"{split_name.upper()} SET VERIFICATION")
        print(f"{'='*70}")
        
        # Basic info
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns[:3])} ... {list(df.columns[-3:-1])} + [Run_ID, Target]")
        
        # Target distribution
        print(f"\nTarget Distribution:")
        target_counts = df['Target'].value_counts().sort_index()
        print(target_counts)
        
        # Calculate percentages
        print(f"\nTarget Percentages:")
        target_pct = (target_counts / len(df) * 100).round(2)
        print(target_pct)
        
        # Unique runs
        unique_runs = df['Run_ID'].nunique()
        print(f"\nUnique Runs: {unique_runs}")
        
        # Check healthy vs faulty ratio
        healthy_count = (df['Target'] == 0).sum()
        faulty_count = (df['Target'] != 0).sum()
        print(f"\nHealthy (Class 0) samples: {healthy_count:,} ({healthy_count/len(df)*100:.2f}%)")
        print(f"Faulty (Class 1-28) samples: {faulty_count:,} ({faulty_count/len(df)*100:.2f}%)")
        
        # Expected ratio check
        expected_healthy_pct = 30.0
        actual_healthy_pct = healthy_count / len(df) * 100
        
        if abs(actual_healthy_pct - expected_healthy_pct) < 2.0:
            print(f"✓ Healthy/Faulty ratio is correct (~30% healthy)")
        else:
            print(f"⚠ Warning: Healthy ratio is {actual_healthy_pct:.2f}%, expected ~{expected_healthy_pct}%")
        
        # Show sample rows
        print(f"\n{'-'*70}")
        print("Sample from start (should be healthy - Target = 0):")
        print(f"{'-'*70}")
        print(df[['Run_ID', 'Target']].head(3))
        
        print(f"\n{'-'*70}")
        print("Sample from end (should be faulty - Target = IDV number):")
        print(f"{'-'*70}")
        print(df[['Run_ID', 'Target']].tail(3))
        
        # Check transition point for one run
        sample_run_id = df['Run_ID'].iloc[0]
        sample_run = df[df['Run_ID'] == sample_run_id].reset_index(drop=True)
        
        print(f"\n{'-'*70}")
        print(f"Transition verification for: {sample_run_id}")
        print(f"Rows {self.FAULT_START_INDEX-2}-{self.FAULT_START_INDEX+2} (transition at {self.FAULT_START_INDEX}):")
        print(f"{'-'*70}")
        print(sample_run.loc[self.FAULT_START_INDEX-2:self.FAULT_START_INDEX+2, ['Run_ID', 'Target']])
        print()


def main_extract_and_load():
    """
    Example usage: Extract, split, and load TEP data in one pipeline.
    """
    
    # Initialize processor
    source_file = r'C:\path\to\TEP_Mode1.h5'  # UPDATE THIS PATH
    output_dir = r'C:\path\to\output'  # UPDATE THIS PATH
    
    processor = TEPDataProcessor(source_file, random_seed=42)
    
    # Step 1: Extract and split data
    train_file, val_file, test_file = processor.extract_and_split_data(output_dir)
    
    # Step 2: Load and label data
    train_df = processor.load_and_label_data(train_file, split_name='train')
    val_df = processor.load_and_label_data(val_file, split_name='validation')
    test_df = processor.load_and_label_data(test_file, split_name='test')
    
    # Step 3: Verify data
    processor.verify_dataframe(train_df, 'Train')
    processor.verify_dataframe(val_df, 'Validation')
    processor.verify_dataframe(test_df, 'Test')
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main_extract_and_load()