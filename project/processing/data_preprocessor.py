"""

data_conditioning.py

This defines methods that are important for preparing .ARSENL data for processing
"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import yaml
from pathlib import Path
import xarray as xr
import os


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.df1 = None
        self. fname_pkl = None
        self.fname_nc = None
        self.file_path_pkl = None
        self.file_path_nc = None

        # File params
        self.fname = config['file_params']['fname']                        # File name of raw data
        self.data_dir = config['file_params']['data_dir']                  # Directory of raw data
        self.preprocessed_dir = config['file_params']['preprocessed_dir']  # Directory to store preprocessing files

        # System Params
        self.PRF = config['system_params']['PRF']                      # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']          # [s] clock resolution

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # Plot params
        self.tbinsize = config['plot_params']['tbinsize']  # [s] temporal bin size
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.bg_edges = config['plot_params']['bg_edges']  # [m] range background window
        self.dpi = config['plot_params']['dpi']            # dots-per-inch
        self.figsize = config['plot_params']['figsize']    # figure size in inches
        self.use_ylim = config['plot_params']['use_ylim']  # TRUE value activates 'axes.set_ylim' argument
        self.ylim = config['plot_params']['ylim']          # [km] y-axis limits


    def load_data(self):
        generic_fname = Path(self.fname).stem
        self.fname_pkl = generic_fname + '.pkl'
        self.fname_nc = generic_fname + '_preprocessed.nc'
        self.file_path_pkl = Path(self.data_dir + self.preprocessed_dir) / self.fname_pkl
        self.file_path_nc = Path(self.data_dir + self.preprocessed_dir) / self.fname_nc

        if self.file_path_nc.exists():
            print('Preprocessed datafile already exists.')
        else:
            print('Preprocessed datafile not found. Starting process to create it...\n')
            # If pickle file does not exist yet, then create it
            if self.file_path_pkl.exists() == False:
                print('No existing pickle object found.\nCreating pickle object...')
                start = time.time()
                df = pd.read_csv(self.data_dir + self.fname, delimiter=',')
                # df = pd.read_csv(self.data_dir + self.fname, delimiter=',', encoding='latin-1', on_bad_lines='skip')
                outfile = open('{}/{}/{}'.format(self.data_dir, self.preprocessed_dir, self.fname_pkl), 'wb')
                pickle.dump(df, outfile)
                outfile.close()
                print('Finished pickling.\nTime elapsed: {:.2f} s'.format(time.time() - start))

            # Unpickle the data to DataFrame
            print('Pickle file found. Loading...')
            infile = open('{}/{}/{}'.format(self.data_dir, self.preprocessed_dir, self.fname_pkl), 'rb')
            self.df = pickle.load(infile)
            infile.close()
            print('Finished loading.')


    def preprocess(self):
        # Load preprocessed data if exists. Otherwise, preprocess and save out results to .nc file.
        if self.file_path_nc.exists():
            print('\nPreprocessed data file found. Loading data file...')
            ds = xr.open_dataset(self.file_path_nc)
            ranges = ds['ranges']
            shots_time = ds['shots_time']
            print('Preprocessed data file loaded.')
        else:
            print('\nPreprocessed data file not found. Creating file...\nStarting preprocessing...')
            df = self.df
            rollover = df.loc[(df['overflow'] == 1) & (
                    df['channel'] == 63)]  # Clock rollover ("overflow", "channel" = 1,63) Max count is 2^25-1=33554431

            start = time.time()
            # Create new dataframe without rollover events
            df1 = df.drop(rollover.index)  # Remove rollover events
            df1 = df1.reset_index(drop=True)  # Reset indices

            # Identify detection events ('detect') and laser pulse events ('sync')
            detect = df1.loc[
                (df1['overflow'] == 0) & (
                        df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
            sync = df1.loc[
                (df1['overflow'] == 1) & (
                        df1['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)

            # Ignore detections that precede first laser pulse event
            start_idx = sync.index[0]
            detect = detect[detect.index > start_idx]

            # Detection "times" in clock counts. Note each clock count is equivalent to 25 ps
            sync_times = sync['dtime']
            detect_times = detect['dtime']

            counts = np.diff(
                sync.index) - 1  # Number of detections per pulse (subtract 1 since sync event is included in np.diff operation)
            remainder = detect.index[-1] - sync.index[-1]
            counts = np.append(counts, remainder)  # Include last laser shot too
            sync_ref = np.repeat(sync_times,
                                 counts)  # Repeated sync time array that stores the corresponding timestamp of the laser event. Each element has a corresponding detection event.
            shots_ref = np.repeat(np.arange(len(sync)),
                                  counts)  # Repeated laser index array. Each element refers to the laser shot number that corresponds to a detection event.
            shots_time = shots_ref / self.PRF  # [s] Equivalent time for each shot TODO: fix PRF estimation and use sync timestamps

            # Convert detection absolute timestamps to relative timestamps
            detect_times_rel = detect_times.to_numpy() - sync_ref.to_numpy()

            # Handle rollover events. Add the clock rollover value to any negative timestamps.
            # A rollover is where the timestamps cycle back to 1 after the clock has reached 2^25-1.
            # This is because if detections occurred between a rollover and sync event, then corresponding "detect_time_rel" element will be negative.
            rollover_idx = np.where(detect_times_rel < 0)[0]
            detect_times_rel[rollover_idx] += self.unwrap_modulo

            # Convert to flight times and range
            flight_times = detect_times_rel * self.clock_res  # [s] counts were in 25 ps increments
            ranges = flight_times * self.c / 2  # [m]

            # Save preprocessed data to netCDF
            preprocessed_data = xr.Dataset(
                data_vars=dict(
                    ranges=ranges,
                    shots_time=shots_time
                )
            )

            preprocessed_data.to_netcdf(os.path.join((self.data_dir + self.preprocessed_dir), self.fname_nc))

            print('Finished preprocessing. File created.\nTime elapsed: {:.1f} seconds'.format(time.time() - start))

            self.df1 = df1

        return {
            'ranges': ranges,
            'shots_time': shots_time
        }

    def gen_histogram(self, preprocessed_results):
        # Load preprocessed data
        ranges = preprocessed_results['ranges']
        shots_time = preprocessed_results['shots_time']

        print('\nStarting to generate histogram...')
        start = time.time()
        tbins = np.arange(0, shots_time[-1], self.tbinsize)  # [s]
        rbins = np.arange(0, self.c / 2 / self.PRF, self.rbinsize)  # [m]

        H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
        H = H.T  # flip axes
        flux = H / (self.rbinsize / self.c * 2) / (
                self.tbinsize * self.PRF)  # [Hz] Backscatter flux $\Phi = n/N/(\Delta t)$, where $\Phi$ is flux, $n$ is photon counts, $N$ is laser shots number, and $\Delta t$ is range resolution.

        # Estimate background flux
        bg_edges_idx = [np.argmin(np.abs(rbins - self.bg_edges[0])), np.argmin(np.abs(rbins - self.bg_edges[1]))]
        bg_flux = np.mean(flux[bg_edges_idx[0]:bg_edges_idx[1], :])
        flux_bg_sub = flux - bg_flux  # [Hz] flux with background subtracted

        print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

        return {
            't_binedges': t_binedges,
            'r_binedges': r_binedges,
            'flux_bg_sub': flux_bg_sub,
            'bg_flux': bg_flux
        }

    def plot_histogram(self, histogram_results):
        # Processed data
        flux_bg_sub = histogram_results['flux_bg_sub']  # [Hz] backscatter flux
        bg_flux = histogram_results['bg_flux']  # [Hz] background flux
        t_binedges = histogram_results['t_binedges']  # [s] temporal bin edges
        r_binedges = histogram_results['r_binedges']  # [m] range bin edges

        # Start plotting
        print('\nStarting to generate plot...')
        start = time.time()
        fig = plt.figure(dpi=self.dpi, figsize=(self.figsize[0], self.figsize[1]))
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_bg_sub, cmap='viridis',
                             norm=LogNorm(vmin=bg_flux, vmax=flux_bg_sub.max()))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        fig.suptitle('CoBaLT Backscatter Flux')
        ax.set_title('Scale {:.1f} m x {:.2f} s'.format(self.rbinsize, self.tbinsize))
        if self.use_ylim:
            ax.set_ylim(self.ylim)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Flux [Hz]')
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time()-start))
        plt.show()


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dp = DataPreprocessor(config)
    dp.load_data()
    preprocessed_results = dp.preprocess()
    histogram_results = dp.gen_histogram(preprocessed_results)
    dp.plot_histogram(histogram_results)


if __name__ == '__main__':
    main()
