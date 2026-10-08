import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time

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
process = True
histogram = True
plot_hist = False
histogram_dead_correct = False  # set true to use Mueller-corrected flux
two_dim = True  # set true to process individual histogram profiles one time bin at a time
degree_start = 15
degree_end = 22
two_dim_tbin = 1  # [s]
plot_af_hist = False
plot_fits = False

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
    if scatter:
        dpl.plot_time_tag_scatter(ranges, shots_time, timestamp, low_gain, generic_fname)

    if histogram:
        gh = GenerateHistogram(config)
        r_binedges, t_binedges, flux, H = gh.gen_histogram(ranges, shots_time, low_gain)

        if histogram_dead_correct:
            deadtime = gh.deadtime_lg if low_gain else gh.deadtime_hg  # [s]
            flux = flux / (1 - flux * deadtime)  # [Hz]

        if plot_hist:
            dpl.plot_histogram(flux, t_binedges, r_binedges, timestamp, low_gain, generic_fname)

        if process:
            if two_dim:
                time_bnds = np.arange(t_binedges[0], t_binedges[-1], two_dim_tbin)

                start = time.time()
                lamb_dead = None
                r_centers_trim = None
                for i, (t_start, t_end) in enumerate(zip(time_bnds[:-1], time_bnds[1:])):
                    i_start = np.searchsorted(t_binedges, t_start, side="left")
                    i_end = np.searchsorted(t_binedges, t_end, side="right")

                    H_subset = H[:, i_start:i_end-1]
                    t_binedges_subset = t_binedges[i_start:i_end]
                    H_train, H_val = bootstrap(H_subset)

                    dp = DeadtimeProcessing(config)
                    results =  dp.optimize_complexity(
                        t_binedges_subset,
                        r_binedges,
                        H_train,
                        H_val,
                        low_gain,
                        degree_start,
                        degree_end,
                        plot_af_hist,
                        plot_fits
                    )

                    if i == 0:
                        r_centers_trim = results['r_centers_trim']
                        lamb_dead = np.zeros((len(r_centers_trim), len(time_bnds)-1))

                    lamb_dead[:, i] = results['lamb_out_dead_best']

                print('Total time elapsed: {:.2f} s'.format(time.time() - start))

                r_binedges_trim = np.concatenate([
                    [r_centers_trim[0] - np.diff(r_centers_trim)[0] / 2],
                    (r_centers_trim[:-1] + r_centers_trim[1:]) / 2,
                    [r_centers_trim[-1] + np.diff(r_centers_trim)[-1] / 2]
                ])

                fig, ax = plt.subplots(figsize=(8, 4))

                pcm = ax.pcolormesh(
                    time_bnds,
                    r_binedges_trim/1e3,
                    lamb_dead/1e6,
                    shading="flat"
                )

                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Range [km]")
                fig.colorbar(pcm, ax=ax, label="Flux [MHz]")

                plt.show()

                quit()
            else:
                H_train, H_val = bootstrap(H)

                dp = DeadtimeProcessing(config)
                results = dp.optimize_complexity(
                    t_binedges,
                    r_binedges,
                    H_train,
                    H_val,
                    low_gain,
                    degree_start,
                    degree_end,
                    plot_af_hist,
                    plot_fits
                )
                quit()


if __name__ == "__main__":
    main()