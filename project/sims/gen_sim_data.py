import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import yaml
import time

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

cwd = os.getcwd()
dirLib = cwd + r'/../utils'
if dirLib not in sys.path:
    sys.path.append(dirLib)

from generate_sim_data_utils import gen_sim_data

# Generate gaussian function
def gaussian(x, A, mu, sigma, b):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

home = str(Path.home())
save_dir = home + r'\OneDrive - UCB-O365\ARSENL\Experiments\Cloud Measurements\Sims\deadtime_fitting_tests\preprocessed_data'

class GenerateData:
    def __init__(self, config):
        self.fname = None
        self.lamb = None

        self.c = config['constants']['c']  # [m/s] speed of light
        self.deadtime = config['system_params']['deadtime_hg']  # [s] high-gain detector deadtime
        self.PRF = config['system_params']['PRF']  # [Hz] laser rep rate
        self.r_sim_min, self.r_sim_max = config['sim_params']['sim_ylim']  # [km] range window
        self.r_plot_min, self.r_plot_max = config['plot_params']['ylim']  # [km]
        self.t_min, self.t_max = config['plot_params']['xlim']  # [s]
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.tbinsize = config['plot_params']['tbinsize']  # [s] time bin size

        self.Nshot = int(config['sim_params']['Nshot'])  # number of laser shots
        self.A = config['sim_params']['A']  # [Hz] Gaussian amplitude flux
        self.b = config['sim_params']['b']  # [Hz] background flux
        self.mu = config['sim_params']['mu']  # [m] Gaussian center
        self.sigma = config['sim_params']['sigma']  # [m] Gaussian stdev
        self.laser_pulse_width = config['sim_params']['laser_pulse_width']  # [s] laser pulse width
        self.wrap_deadtime = False

    def generate_data(self):
        r = np.arange(self.r_sim_min*1e3, self.r_sim_max*1e3 + self.rbinsize, self.rbinsize)  # [m] range axis

        r_t = r / self.c * 2  # [s] range axis in time
        mu_t = self.mu / self.c * 2  # [s] center of guassian in time
        sigma_t = self.sigma / self.c * 2  # [s] spread of gaussian in time
        dr_t = self.rbinsize / self.c * 2  # [s] range bin resolution in time

        r_t_min = r_t[0]  # [s] beginning of range window to shift
        r_t_shifted = r_t - r_t_min  # [s] shift the time axis
        mu_t_shifted = mu_t - r_t_min  # [s] shift the center of the Gaussian
        self.lamb = gaussian(r_t_shifted, self.A, mu_t_shifted, sigma_t, self.b)

        start = time.time()
        # Generate simulated data
        sim_results = gen_sim_data(photon_rate_arr=self.lamb,
                                   t_sim_bins = r_t_shifted,
                                   tD = self.deadtime,
                                   Nshot = self.Nshot,
                                   wrap_deadtime=self.wrap_deadtime,
                                   )
        print('Simulated Data generated. Time elapsed: {:.1f} s'.format(time.time() - start))

        time_tag_idx = sim_results['det_idx']  # detected time tag index
        true_time_tag_idx = sim_results['phot_idx']  # incident photon time tag index
        sync_idx = sim_results['sync_idx']  # laser sync events
        time_tag = sim_results['det_events']  # detection time tags
        true_time_tag = sim_results['phot_events']  # incident photon time tags
        time_tag_sync_idx = sim_results['det_sync_idx']  # sync index for detections
        true_time_tag_sync_idx = sim_results['phot_sync_idx']  # sync index for incident photons
        t_sim_bins = sim_results['t_sim_bins']

        dr_t = np.diff(t_sim_bins)[0]  # [s]
        time_tag += r_t_min / dr_t  # shift back to actual time values (in units clock counts)
        true_time_tag += r_t_min / dr_t  # shift back to actual time values (in units clock counts)

        # Save simulated data to netCDF
        sim_data = xr.Dataset(data_vars=dict(
            time_tag=(['time_tag_index'], time_tag),
            time_tag_sync_index=(['time_tag_index'], time_tag_sync_idx),
            true_time_tag=(['true_time_tag_index'], true_time_tag),
            true_time_tag_sync_index=(['true_time_tag_index'], true_time_tag_sync_idx),
            laser_pulse_width=self.laser_pulse_width,
            target_time=mu_t,
            target_amplitude=self.A,
            background=self.b,
            dt_sim=dr_t,
            time_axis=r_t
        ),
            coords=dict(
                sync_index=(['sync_index'], sync_idx)
            )
        )

        self.fname = r'\sim_amp{:.1E}_nshot{:.1E}_width{:.1E}_dt{:.1E}.nc'.format(self.A, self.Nshot,
                                                                             self.laser_pulse_width, dr_t)

        sim_data.to_netcdf(save_dir + self.fname)

        return self.lamb, r

    def load_sim_data(self):
        # Now load data
        ds = xr.open_dataset(save_dir + self.fname)

        cnts = ds.time_tag
        dt = ds.dt_sim

        flight_times = cnts * dt  # [s]
        ranges = flight_times * self.c / 2  # [m]
        shots_time = ds.time_tag_sync_index / self.PRF  # [s]

        # start = time.time()
        tbins = np.arange(self.t_min, self.t_max + self.tbinsize, self.tbinsize)  # [s]
        rbins = np.arange(self.r_sim_min*1e3, self.r_sim_max*1e3 + self.rbinsize, self.rbinsize)  # [m]

        # Generate histogram
        H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
        H = H.T  # flip axes
        flux = H / (self.rbinsize / self.c * 2) / (self.tbinsize * self.PRF)  # [Hz] Backscatter flux

        r_centers = r_binedges[:-1] + self.rbinsize / 2  # [m]
        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111)
        ax.plot(self.lamb[:-1], r_centers, label='Truth')
        ax.plot(flux, r_centers, label='Observed')
        ax.set_xlabel('Flux [Hz]')
        ax.set_ylabel('Range [m]')
        ax.set_title('Simulated Measurements')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return {
            't_binedges': t_binedges,
            'r_binedges': r_binedges,
            'flux_raw': flux,
            'cnts_raw': H
        }


if __name__ == '__main__':
    # Load sim data config params
    config_path = Path(__file__).resolve().parent.parent / "config" / "sim_deadtime_fitting_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gd = GenerateData(config)
    gd.generate_data()
    histogram_results = gd.load_sim_data()










