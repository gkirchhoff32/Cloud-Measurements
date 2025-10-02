"""
run_preprocessor.py

script to run the preprocessing routine from "data_preprocessor.py" which converts the timestamps from the binary format .ARSENL file to ranges
and laser shots. Plot the data in scatter or histogram format.

09.19.2025
"""

import yaml
from pathlib import Path

from processing.data_preprocessor_v2 import DataProcessor

def main():
    """
    Load .ARSENL file --> preprocess to netcdf file --> generate and display histogram or display scatter plot
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dp = DataProcessor(config)
    dp.run()


if __name__ == '__main__':
    main()
