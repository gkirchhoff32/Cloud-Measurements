import sys
from pathlib import Path
import yaml
import numpy as np

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
scatter = False
process = False
histogram = True
histogram_dead_correct = True  # set true to use Mueller-corrected flux
degree_start = 2
degree_end = 8

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

    step = 200  # [s]
    tmin = 0  # [s]
    tmax = 1800  # [s]
    bnds = np.arange(tmin, tmax + step, step)
    for i in range(len(bnds)-1):
        if use_sim:
            gsd = GenerateSimData(config)
            gsd.write_sim_data()

            ncl = netCDFLoader(config, use_sim)
            ranges, shots_time, generic_fname = ncl.load_sim_data()
        else:
            dpp = DataPreprocessor(config)
            preprocess_path, generic_fname, low_gain, timestamp = dpp.write_ARSENL_netCDF()

            ncl = netCDFLoader(config, use_sim)
            ncl.xlim = [bnds[i], bnds[i+1]]
            ranges, shots_time = ncl.load_chunks(preprocess_path, generic_fname)

        dpl = DataPlotter(config)
        dpl.xlim = [bnds[i], bnds[i + 1]]
        if scatter:
            dpl.plot_time_tag_scatter(ranges, shots_time, timestamp, low_gain, generic_fname)

        if histogram:
            gh = GenerateHistogram(config)
            gh.xlim = [bnds[i], bnds[i + 1]]
            r_binedges, t_binedges, flux, H = gh.gen_histogram(ranges, shots_time, low_gain)

            if histogram_dead_correct:
                deadtime = gh.deadtime_lg if low_gain else gh.deadtime_hg  # [s]
                flux = flux / (1 - flux * deadtime)  # [Hz]

                # flux *= (10 ** (0.3 - 0.1))  # [Hz] Adjust to OD0.3 and OD0.1 difference

            dpl.plot_histogram(flux, t_binedges, r_binedges, timestamp, low_gain, generic_fname)


if __name__ == "__main__":
    main()