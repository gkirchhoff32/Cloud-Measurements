import sys
from pathlib import Path
import yaml

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from processing.data_preprocessor import DataPreprocessor
from io_utils.netCDF_loader import netCDFLoader
from visualizations.data_plotter import DataPlotter

use_sim = False

def main():
    run()

def run():
    if use_sim:
        config_path = Path(__file__).resolve().parent.parent / "config" / "sim_deadtime_fitting_config.yaml"
    else:
        config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if use_sim:
        quit()
    else:
        dpp = DataPreprocessor(config)
        preprocess_path, generic_fname, low_gain, timestamp = dpp.write_ARSENL_netCDF()

        ncl = netCDFLoader(config)
        ranges, shots_time = ncl.load_chunks(preprocess_path, generic_fname)

        dpl = DataPlotter(config)
        dpl.plot_time_tag_scatter(ranges, shots_time, timestamp, low_gain, generic_fname)

        quit()


if __name__ == "__main__":
    main()