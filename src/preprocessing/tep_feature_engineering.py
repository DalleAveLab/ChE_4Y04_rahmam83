"""
Tennessee Eastman Process (TEP) Feature Engineering Module

This module handles feature engineering steps:
1. Dropping zero-variance features
2. Dropping analyzer measurement features (unrealistic for real-time application)

"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from typing import List, Tuple, Optional
import pickle


class TEPFeatureEngineer:
    """
    Handles feature engineering for Tennessee Eastman Process data.
    
    Steps:
    1. Drop zero-variance features
    2. Drop analyzer measurements (unrealistic for real-time use)
    """
    
    def __init__(self):
        """Initialize the Feature Engineer."""
        self.zero_var_features = []
        self.analyzer_features = []
        self.features_to_drop = []
        self.kept_features = []
        
    def identify_analyzer_features(self, df: pd.DataFrame, 
                                   analyzer_list: Optional[List[str]] = None) -> List[str]:
        """
        Identify analyzer measurement features to drop based on explicit list.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        analyzer_list : List[str], optional
            Explicit list of column names to drop.
            These should be EXACT column names from your data.
        
        Returns:
        --------
        List[str]
            List of analyzer feature column names found in the DataFrame
        """
        
        # Get all feature columns (excluding Run_ID and Target)
        feature_cols = set(df.columns) - {'Run_ID', 'Target'}
        
        analyzer_features = []
        not_found = []
        
        if analyzer_list:
            print(f"Using explicit feature list from config ({len(analyzer_list)} features specified)")
            
            for col_name in analyzer_list:
                if col_name in feature_cols:
                    analyzer_features.append(col_name)
                else:
                    not_found.append(col_name)
            
            if not_found:
                print(f"\n⚠ Warning: {len(not_found)} features from config not found in DataFrame:")
                for feat in not_found:
                    print(f"  - '{feat}'")
                print(f"\nAvailable columns include: {list(feature_cols)[:5]}...")
        
        else:
            print("No explicit feature list provided - no analyzer features will be dropped")
        
        self.analyzer_features = analyzer_features
        
        print(f"\n{'='*70}")
        print("ANALYZER FEATURES TO DROP")
        print(f"{'='*70}")
        
        if len(analyzer_features) > 0:
            print(f"Will drop {len(analyzer_features)} features:")
            for i, feat in enumerate(analyzer_features, 1):
                print(f"  {i}. {feat}")
        else:
            print("  (None specified or found)")
        
        print(f"{'='*70}\n")
        
        return analyzer_features
    
    def identify_zero_variance_features(self, df: pd.DataFrame, threshold: float = 0.0) -> List[str]:
        """
        Identify features with zero or near-zero variance.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame (should be training data only)
        threshold : float
            Variance threshold (default: 0.0 for zero variance)
        
        Returns:
        --------
        List[str]
            List of zero-variance feature column names
        """
        
        # Get all feature columns (excluding Run_ID, Target, and Time)
        feature_cols = [col for col in df.columns 
                       if col not in ['Run_ID', 'Target', 'Time']]
        
        # Calculate variance for each feature
        variances = df[feature_cols].var()
        
        # Identify zero-variance features
        zero_var_features = variances[variances <= threshold].index.tolist()
        
        self.zero_var_features = zero_var_features
        
        print(f"\n{'='*70}")
        print("ZERO-VARIANCE FEATURES IDENTIFIED")
        print(f"{'='*70}")
        
        if len(zero_var_features) == 0:
            print("✓ No zero-variance features found (all features have variation)")
        else:
            print(f"Found {len(zero_var_features)} zero-variance features:")
            for i, feat in enumerate(zero_var_features, 1):
                print(f"  {i}. {feat} (variance: {variances[feat]:.6f})")
        
        print(f"{'='*70}\n")
        
        return zero_var_features
    
    def fit(self, train_df: pd.DataFrame, drop_analyzers: bool = True, 
            drop_zero_var: bool = True, variance_threshold: float = 0.0,
            analyzer_list: Optional[List[str]] = None):
        """
        Fit the feature engineer on training data to identify features to drop.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training DataFrame
        drop_analyzers : bool
            Whether to drop analyzer features
        drop_zero_var : bool
            Whether to drop zero-variance features
        variance_threshold : float
            Variance threshold for dropping features
        analyzer_list : List[str], optional
            Explicit list of analyzer feature names to drop.
            If None, will search for common patterns.
        """
        
        print(f"\n{'='*70}")
        print("FITTING FEATURE ENGINEER ON TRAINING DATA")
        print(f"{'='*70}")
        print(f"Original features: {train_df.shape[1] - 2}")  # -2 for Run_ID, Target
        print(f"Drop analyzers: {drop_analyzers}")
        print(f"Drop zero-variance: {drop_zero_var}")
        print(f"Variance threshold: {variance_threshold}")
        print(f"{'='*70}")
        
        # Identify features to drop
        features_to_drop = []
        
        if drop_zero_var:
            zero_var = self.identify_zero_variance_features(train_df, variance_threshold)
            features_to_drop.extend(zero_var)
        
        if drop_analyzers:
            analyzers = self.identify_analyzer_features(train_df, analyzer_list)
            features_to_drop.extend(analyzers)
        
        # Remove duplicates
        self.features_to_drop = list(set(features_to_drop))
        
        # Identify kept features
        all_features = [col for col in train_df.columns 
                       if col not in ['Run_ID', 'Target']]
        self.kept_features = [col for col in all_features 
                             if col not in self.features_to_drop]
        
        print(f"\n{'='*70}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*70}")
        print(f"Total features to drop: {len(self.features_to_drop)}")
        print(f"  - Zero-variance: {len(self.zero_var_features)}")
        print(f"  - Analyzers: {len(self.analyzer_features)}")
        print(f"Features remaining: {len(self.kept_features)}")
        print(f"{'='*70}\n")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by dropping identified features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with features dropped
        """
        
        if not self.features_to_drop:
            print("⚠ Warning: No features identified to drop. Call fit() first.")
            return df.copy()
        
        # Drop the features
        df_transformed = df.drop(columns=self.features_to_drop, errors='ignore')
        
        print(f"Transformed: {df.shape[1]} → {df_transformed.shape[1]} columns")
        
        return df_transformed
    
    def fit_transform(self, train_df: pd.DataFrame, drop_analyzers: bool = True,
                     drop_zero_var: bool = True, variance_threshold: float = 0.0,
                     analyzer_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training DataFrame
        drop_analyzers : bool
            Whether to drop analyzer features
        drop_zero_var : bool
            Whether to drop zero-variance features
        variance_threshold : float
            Variance threshold for dropping features
        analyzer_list : List[str], optional
            Explicit list of analyzer feature names
        
        Returns:
        --------
        pd.DataFrame
            Transformed training DataFrame
        """
        
        self.fit(train_df, drop_analyzers, drop_zero_var, variance_threshold, analyzer_list)
        return self.transform(train_df)
    
    def save(self, filepath: str):
        """
        Save the feature engineer to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where feature engineer will be saved
        """
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Feature engineer saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """
        Load a feature engineer from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to saved feature engineer file
        
        Returns:
        --------
        TEPFeatureEngineer
            Loaded feature engineer object
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            fe = pickle.load(f)
        print(f"✓ Feature engineer loaded from {filepath}")
        return fe
    
    def get_feature_info(self) -> dict:
        """
        Get information about feature engineering decisions.
        
        Returns:
        --------
        dict
            Dictionary with feature information
        """
        return {
            'total_features_dropped': len(self.features_to_drop),
            'zero_variance_features': self.zero_var_features,
            'analyzer_features': self.analyzer_features,
            'features_to_drop': self.features_to_drop,
            'kept_features': self.kept_features,
            'n_kept_features': len(self.kept_features)
        }


def apply_feature_engineering(train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                              test_df: pd.DataFrame = None,
                              drop_analyzers: bool = True,
                              drop_zero_var: bool = True,
                              variance_threshold: float = 0.0,
                              analyzer_list: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply feature engineering to train/val/test DataFrames.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    val_df : pd.DataFrame
        Validation DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    drop_analyzers : bool
        Whether to drop analyzer features
    drop_zero_var : bool
        Whether to drop zero-variance features
    variance_threshold : float
        Variance threshold for dropping features
    analyzer_list : List[str], optional
        Explicit list of analyzer feature names to drop
    save_path : str, optional
        Path to save the feature engineer object
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Transformed (train, val, test) DataFrames
    """
    
    print("\n" + "="*70)
    print("APPLYING FEATURE ENGINEERING")
    print("="*70 + "\n")
    
    # Initialize feature engineer
    fe = TEPFeatureEngineer()
    
    # Fit on training data
    train_transformed = fe.fit_transform(
        train_df, 
        drop_analyzers=drop_analyzers,
        drop_zero_var=drop_zero_var,
        variance_threshold=variance_threshold,
        analyzer_list=analyzer_list
    )
    
    # Transform validation and test data (val is optional)
    if val_df is not None:
        print("\nTransforming validation set...")
        val_transformed = fe.transform(val_df)
    else:
        val_transformed = None

    print("Transforming test set...")
    test_transformed = fe.transform(test_df)
    
    # Save if path provided
    if save_path:
        fe.save(save_path)
    
    print("\n" + "="*70)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"\nFinal feature count: {len(fe.kept_features)}")
    print(f"Features dropped: {len(fe.features_to_drop)}")
    print("="*70 + "\n")
    
    return train_transformed, val_transformed, test_transformed


if __name__ == "__main__":
    # Example usage
    print("TEP Feature Engineering Module")
    print("Import this module to use feature engineering functions")