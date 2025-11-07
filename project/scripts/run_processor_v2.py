"""
run_preprocessor.py

script to run the preprocessing routine from "data_preprocessor.py" which converts the timestamps from the binary format .ARSENL file to ranges
and laser shots. Plot the data in scatter or histogram format.

09.19.2025
"""

import sys
import yaml
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from processing.data_preprocessor_v2 import DataPreprocessor

def main():
    """
    Load .ARSENL file --> preprocess to netcdf file --> generate and display histogram or display scatter plot
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dp = DataPreprocessor(config)
    dp.run()


if __name__ == '__main__':
    main()
