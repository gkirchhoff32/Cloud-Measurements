import sys
from pathlib import Path
import yaml

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from processing.data_preprocessor import DataPreprocessor
from io_utils.netCDF_loader import netCDFLoader
from visualizations.plotter import DataPlotter
from processing.generate_histogram import GenerateHistogram
from processing.deadtime_processing import DeadtimeProcessing
from simulation.gen_sim_data import GenerateSimData
from utils.array_utils import bootstrap

use_sim = False
process = True
degree_start = 2
degree_end = 30

def main():
    run()

def run():
    low_gain = False
    timestamp = None
    if use_sim:
        config_path = Path(__file__).resolve().parent.parent / "config" / "sim_deadtime_fitting_config.yaml"
    else:
        config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if use_sim:
        gsd = GenerateSimData(config)
        gsd.write_sim_data()

        ncl = netCDFLoader(config, use_sim)
        ranges, shots_time, generic_fname = ncl.load_sim_data()
    else:
        dpp = DataPreprocessor(config)
        preprocess_path, generic_fname, low_gain, timestamp = dpp.write_ARSENL_netCDF()

        ncl = netCDFLoader(config, use_sim)
        ranges, shots_time = ncl.load_chunks(preprocess_path, generic_fname)

    dpl = DataPlotter(config)
    # dpl.plot_time_tag_scatter(ranges, shots_time, timestamp, low_gain, generic_fname)

    gh = GenerateHistogram(config)
    r_binedges, t_binedges, flux, H = gh.gen_histogram(ranges, shots_time, low_gain)
    dpl.plot_histogram(flux, t_binedges, r_binedges, timestamp, low_gain, generic_fname)

    if process:
        H_train, H_val = bootstrap(H)

        dp = DeadtimeProcessing(config)
        # flux_bin_est, r_binedges_trim = dp.binwise_correction(flux, r_binedges, t_binedges, H, low_gain)
        # dpl.plot_histogram(flux_bin_est, t_binedges, r_binedges_trim, timestamp, low_gain, generic_fname)

        # dp.deadtime_fitting(H, r_binedges, t_binedges, low_gain)
        dp.optimize_complexity(t_binedges, r_binedges, H_train, H_val, low_gain, degree_start, degree_end)
        quit()


if __name__ == "__main__":
    main()