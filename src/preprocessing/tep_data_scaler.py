"""
Tennessee Eastman Process (TEP) Data Scaling Module

This module provides utilities for scaling TEP data using StandardScaler.
Time, Run_ID, and Target columns are kept unscaled.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Optional


class TEPDataScaler:
    """
    Handles scaling of Tennessee Eastman Process data.
    
    Fits a StandardScaler on training data and applies it to train/val/test sets.
    Time column is kept unscaled. Run_ID and Target are preserved as-is.
    """
    
    def __init__(self):
        """Initialize the TEP Data Scaler."""
        self.scaler = None
        self.feature_columns = None
    
    def fit_scaler_dataframe(self, train_df: pd.DataFrame) -> StandardScaler:
        """
        Fit StandardScaler on training DataFrame (excluding Time, Run_ID, Target).
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training DataFrame
        
        Returns:
        --------
        StandardScaler
            Fitted scaler object
        """
        print("="*70)
        print("FITTING SCALER ON TRAINING DATA (DataFrame)")
        print("="*70)
        print("NOTE: Time, Run_ID, and Target columns will NOT be scaled\n")
        
        # Identify feature columns (exclude Time, Run_ID, Target)
        self.feature_columns = [col for col in train_df.columns 
                               if col not in ['Time', 'Run_ID', 'Target']]
        
        print(f"Feature columns to scale: {len(self.feature_columns)}")
        print(f"  Excluded: Time, Run_ID, Target")
        
        # Extract feature data
        feature_data = train_df[self.feature_columns].values
        
        print(f"\nTraining data shape: {feature_data.shape}")
        print(f"Memory usage: {feature_data.nbytes / 1024**2:.2f} MB")
        
        # Fit the scaler
        print(f"\nFitting StandardScaler on {len(self.feature_columns)} features...")
        self.scaler = StandardScaler()
        self.scaler.fit(feature_data)
        
        print(f"✓ Scaler fitted successfully")
        print(f"  - Mean shape: {self.scaler.mean_.shape}")
        print(f"  - Std shape: {self.scaler.scale_.shape}")
        print(f"  - Sample means (first 5): {self.scaler.mean_[:5]}")
        print(f"  - Sample stds (first 5): {self.scaler.scale_[:5]}\n")
        
        return self.scaler
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted scaler.
        Time, Run_ID, and Target columns are preserved as-is.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit_scaler_dataframe() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Call fit_scaler_dataframe() first.")
        
        # Create a copy to avoid modifying original
        df_scaled = df.copy()
        
        # Scale only the feature columns
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_scaled
    
    def verify_scaling_dataframe(self, df: pd.DataFrame, split_name: str = "Data"):
        """
        Verify that scaling was applied correctly to a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Scaled DataFrame to verify
        split_name : str
            Name of the split for display
        """
        print(f"\nVerifying scaling for: {split_name}")
        
        # Check Time column (should NOT be scaled)
        if 'Time' in df.columns:
            time_col = df['Time']
            print(f"  Time column - should be UNSCALED:")
            print(f"    Min: {time_col.min():.2f}, Max: {time_col.max():.2f}")
            print(f"    Mean: {time_col.mean():.2f}, Std: {time_col.std():.2f}")
            
            if 0 <= time_col.min() and time_col.max() < 200:
                print(f"    ✓ Time column appears unscaled (reasonable values)")
            else:
                print(f"    ⚠ Warning: Time column may have unexpected values")
        
        # Check scaled features
        if self.feature_columns:
            scaled_features = df[self.feature_columns]
            sample_mean = scaled_features.mean().mean()
            sample_std = scaled_features.std().mean()
            
            print(f"\n  Scaled features ({len(self.feature_columns)} columns):")
            print(f"    Mean (average across features): {sample_mean:.6f}")
            print(f"    Std (average across features): {sample_std:.6f}")
            print(f"    Expected: mean ≈ 0, std ≈ 1")
            
            if abs(sample_mean) < 0.1 and abs(sample_std - 1.0) < 0.3:
                print(f"    ✓ Scaling appears correct\n")
            else:
                print(f"    ⚠ Warning: scaling may not be optimal\n")
    
    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where scaler will be saved
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """
        Load a fitted scaler from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to saved scaler file
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")
        return self.scaler


def scale_dataframes(train_df: pd.DataFrame,
                     val_df: pd.DataFrame = None,
                     test_df: pd.DataFrame = None,
                     save_scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TEPDataScaler]:
    """
    Scale train/val/test DataFrames using StandardScaler.
    
    Fits scaler on training data and applies to all splits.
    Time, Run_ID, and Target columns are preserved as-is.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    val_df : pd.DataFrame
        Validation DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    save_scaler_path : str, optional
        Path to save the fitted scaler
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TEPDataScaler]
        Scaled (train, val, test) DataFrames and the scaler object
    """
    
    print("\n" + "="*70)
    print("SCALING DATAFRAMES")
    print("="*70 + "\n")
    
    # Initialize scaler
    scaler_obj = TEPDataScaler()
    
    # Fit on training data
    print("Step 1: Fitting scaler on training data...")
    scaler_obj.fit_scaler_dataframe(train_df)
    
    # Transform all datasets
    print("Step 2: Transforming datasets...")
    print("  Transforming training data...")
    train_scaled = scaler_obj.transform_dataframe(train_df)
    
    if val_df is not None:
        print("  Transforming validation data...")
        val_scaled = scaler_obj.transform_dataframe(val_df)
    else:
        val_scaled = None

    print("  Transforming test data...")
    test_scaled = scaler_obj.transform_dataframe(test_df)

    # Verify scaling
    print("\nStep 3: Verifying scaling...")
    scaler_obj.verify_scaling_dataframe(train_scaled, "Training Set")
    if val_scaled is not None:
        scaler_obj.verify_scaling_dataframe(val_scaled, "Validation Set")
    scaler_obj.verify_scaling_dataframe(test_scaled, "Test Set")
    
    # Save scaler if path provided
    if save_scaler_path:
        scaler_obj.save_scaler(save_scaler_path)
    
    print("="*70)
    print("✓ SCALING COMPLETE")
    print("="*70 + "\n")
    
    return train_scaled, val_scaled, test_scaled, scaler_obj


if __name__ == "__main__":
    print("TEP Data Scaler Module")
    print("Import this module to use scaling functions")