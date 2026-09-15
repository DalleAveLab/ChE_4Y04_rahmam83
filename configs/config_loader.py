"""
Configuration Loader for TEP Pipeline

Loads settings from config.yaml file.

"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """
    Configuration loader for TEP pipeline.
    
    Loads settings from config.yaml and provides easy access to parameters.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to config.yaml file (default: 'config.yaml' in current directory)
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file or specify the correct path."
            )
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Validate required fields
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration fields are present."""
        required_sections = ['data', 'random_seed', 'feature_engineering', 'save_options']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: '{section}'")
        
        # Validate data paths section
        if 'raw_source' not in self._config['data']:
            raise ValueError("Missing required field: data.raw_source")
        if 'processed_base_dir' not in self._config['data']:
            raise ValueError("Missing required field: data.processed_base_dir")
    
    @property
    def raw_source(self) -> str:
        """Path to raw source TEP_Mode1.h5 file."""
        return self._config['data']['raw_source']
    
    @property
    def _experiment_suffix(self) -> str:
        """Suffix derived from split settings, e.g. 'N50_tr30_v10_te10'."""
        splits = self._config.get('splits', {})
        total = splits.get('total_runs', 50)
        train = splits.get('train_runs', 30)
        val   = splits.get('val_runs', 10)
        test  = splits.get('test_runs', 10)
        return f"N{total}_tr{train}_v{val}_te{test}"

    @property
    def output_dir(self) -> str:
        """Output directory for processed data, auto-derived from split settings."""
        base = self._config['data']['processed_base_dir']
        return f"{base}_{self._experiment_suffix}"
    
    @property
    def random_seed(self) -> int:
        """Random seed for reproducibility."""
        return self._config.get('random_seed', 42)
    
    @property
    def drop_analyzers(self) -> bool:
        """Whether to drop analyzer features."""
        return self._config['feature_engineering'].get('drop_analyzers', True)
    
    @property
    def drop_zero_variance(self) -> bool:
        """Whether to drop zero-variance features."""
        return self._config['feature_engineering'].get('drop_zero_variance', True)
    
    @property
    def variance_threshold(self) -> float:
        """Variance threshold for feature selection."""
        return self._config['feature_engineering'].get('variance_threshold', 0.0)
    
    @property
    def analyzer_features(self) -> list:
        """List of analyzer features to drop."""
        return self._config['feature_engineering'].get('analyzer_features', [])
    
    @property
    def save_dataframes(self) -> bool:
        """Whether to save final DataFrames."""
        return self._config['save_options'].get('save_dataframes', True)
    
    @property
    def save_csv_sample(self) -> bool:
        """Whether to save CSV sample."""
        return self._config['save_options'].get('save_csv_sample', True)
    
    @property
    def csv_sample_rows(self) -> int:
        """Number of rows in CSV sample."""
        return self._config['save_options'].get('csv_sample_rows', 10000)
    
    @property
    def total_runs(self) -> int:
        """Total runs to select per IDV."""
        return self._config.get('splits', {}).get('total_runs', 50)
    
    @property
    def train_runs(self) -> int:
        """Number of training runs."""
        return self._config.get('splits', {}).get('train_runs', 30)
    
    @property
    def val_runs(self) -> int:
        """Number of validation runs."""
        return self._config.get('splits', {}).get('val_runs', 10)
    
    @property
    def test_runs(self) -> int:
        """Number of test runs."""
        return self._config.get('splits', {}).get('test_runs', 10)
    
    @property
    def window_size(self) -> int:
        """Number of timesteps per window."""
        return self._config.get("windowing", {}).get("window_size", 5)

    @property
    def stride(self) -> int:
        """Step size between windows."""
        return self._config.get("windowing", {}).get("stride", 1)

    @property
    def save_metadata(self) -> bool:
        """Whether to save window metadata."""
        return self._config.get("windowing", {}).get("save_metadata", True)
    
    @property
    def n_trials(self) -> int:
        return self._config.get('tuning', {}).get('n_trials', 50)

    @property
    def results_dir(self) -> str:
        base = self._config.get('models', {}).get('results_base_dir', 'results')
        return f"{base}_{self._experiment_suffix}"

    @property
    def tuning_search_space(self) -> dict:
        return self._config.get('tuning', {}).get('search_space', {})

    @property
    def batch_size(self) -> int:
        return self._config.get('training', {}).get('batch_size', 256)

    @property
    def max_epochs(self) -> int:
        return self._config.get('training', {}).get('max_epochs', 50)

    @property
    def early_stopping_patience(self) -> int:
        return self._config.get('training', {}).get('early_stopping_patience', 5)

    @property
    def num_workers(self) -> int:
        return self._config.get('training', {}).get('num_workers', 0)
    
    def display(self):
        """Display current configuration."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        print("\nData Paths:")
        print(f"  Raw source:  {self.raw_source}")
        print(f"  Output dir:  {self.output_dir}  (auto-derived from splits)")
        print(f"  Results dir: {self.results_dir}  (auto-derived from splits)")
        print(f"\nRandom seed: {self.random_seed}")
        print(f"\nFeature Engineering:")
        print(f"  Drop analyzers: {self.drop_analyzers}")
        if self.drop_analyzers and self.analyzer_features:
            print(f"  Analyzer features to drop: {len(self.analyzer_features)}")
            for feat in self.analyzer_features[:3]:
                print(f"    - {feat}")
            if len(self.analyzer_features) > 3:
                print(f"    ... and {len(self.analyzer_features) - 3} more")
        print(f"  Drop zero-variance: {self.drop_zero_variance}")
        print(f"  Variance threshold: {self.variance_threshold}")
        print(f"\nSave Options:")
        print(f"  Save DataFrames: {self.save_dataframes}")
        print(f"  Save CSV sample: {self.save_csv_sample}")
        print(f"  CSV sample rows: {self.csv_sample_rows}")
        print(f"\nData Splits:")
        print(f"  Total runs per IDV: {self.total_runs}")
        print(f"  Train: {self.train_runs}")
        print(f"  Val: {self.val_runs}")
        print(f"  Test: {self.test_runs}")
        print(f"\nWindowing:")
        print(f"  Window size: {self.window_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Save metadata: {self.save_metadata}")
        print("="*70 + "\n")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports nested keys with dot notation).
        
        Parameters:
        -----------
        key : str
            Configuration key (e.g., 'data.raw_source')
        default : Any
            Default value if key not found
        
        Returns:
        --------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def load_config(config_path: str = 'config.yaml') -> Config:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to config.yaml file
    
    Returns:
    --------
    Config
        Configuration object
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        config.display()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease create a config.yaml file in the current directory.")
