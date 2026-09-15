"""
Helper Script: Identify Analyzer Features

Run this script to see all column names in your data
and identify which ones are analyzer measurements.

"""

import h5py
import yaml
from pathlib import Path


def inspect_tep_data(h5_file_path: str):
    """
    Inspect TEP HDF5 file and display all column names.
    """

    print("\n" + "="*80)
    print(" "*20 + "TEP DATA COLUMN INSPECTOR")
    print("="*80)
    print(f"\nFile: {h5_file_path}\n")

    try:
        with h5py.File(h5_file_path, 'r') as f:

            print("PROCESSDATA LABELS (Main features)")
            print("-"*80)
            processdata_labels = f['Processdata_Labels'][:].astype(str)
            for i, label in enumerate(processdata_labels, 1):
                print(f"  {i:2d}. {label}")

            print(f"\n{'='*80}")
            print("📊 ADDITIONAL_MEAS LABELS (Additional measurements)")
            print("-"*80)
            additional_meas_labels = f['Additional_Meas_Labels'][:].astype(str)
            for i, label in enumerate(additional_meas_labels, 1):
                print(f"  {i:2d}. {label}")

            print("="*80 + "\n")

    except FileNotFoundError:
        print(f"❌ Error: File not found: {h5_file_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def load_h5_path_from_config():
    """Load H5 path from root configs/config.yaml"""

    # Go up from src/utils → project root
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["data"]["raw_source"]


def main():
    """Main entrypoint"""

    h5_file_path = load_h5_path_from_config()
    print(f"\nUsing H5 file from config: {h5_file_path}")

    inspect_tep_data(h5_file_path)


if __name__ == "__main__":
    main()
