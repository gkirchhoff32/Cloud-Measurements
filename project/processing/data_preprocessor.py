"""

data_conditioning.py

This defines methods that are important for preparing .ARSENL data for processing
"""

import numpy as np
import time
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
        self.fname_nc = None
        self.file_path_nc = None
        self.generic_fname = None
        self.img_save_path = None

        # File params
        self.fname = config['file_params']['fname']                        # File name of raw data
        self.data_dir = config['file_params']['data_dir']                  # Directory of raw data
        self.preprocessed_dir = config['file_params']['preprocessed_dir']  # Directory to store preprocessing files
        self.image_dir = config['file_params']['image_dir']                # Directory to save images

        # System Params
        self.PRF = config['system_params']['PRF']                      # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']          # [s] clock resolution

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # Plot params
        self.tbinsize = config['plot_params']['tbinsize']          # [s] temporal bin size
        self.rbinsize = config['plot_params']['rbinsize']          # [m] range bin size
        self.bg_edges = config['plot_params']['bg_edges']          # [m] range background window
        self.dpi = config['plot_params']['dpi']                    # dots-per-inch
        self.figsize = config['plot_params']['figsize']            # figure size in inches
        self.use_ylim = config['plot_params']['use_ylim']          # TRUE value activates 'axes.set_ylim' argument
        self.use_xlim = config['plot_params']['use_xlim']          # TRUE value activates 'axes.set_xlim' argument
        self.ylim = config['plot_params']['ylim']                  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']                  # [s] x-axis limits
        self.histogram = config['plot_params']['histogram']        # Plot histogram if TRUE, else scatter plot
        self.save_img = config['plot_params']['save_img']          # Save images if TRUE
        self.save_dpi = config['plot_params']['save_dpi']          # DPI for saved images
        self.dot_size = config['plot_params']['dot_size']          # Dot size for 'axes.scatter' 's' param
        self.flux_correct = config['plot_params']['flux_correct']  # TRUE will plot flux that has been background subtracted and range corrected


    def preprocess(self):
        self.generic_fname = Path(self.fname).stem
        self.fname_nc = self.generic_fname + '.nc'
        self.file_path_nc = Path(self.data_dir + self.preprocessed_dir) / self.fname_nc

        print('\nPreprocessed data file not found. Creating file...\nStarting preprocessing...')
        # Look at first row to check for headers.
        headers = ['dev', 'sec', 'usec', 'overflow', 'channel', 'dtime', '[uint32_t version of binary data]']
        start_row = 0
        chunk_num = 0
        chunksize = 0.1  # [MB]
        global_start = time.time()

        # First estimate how many rows are approximately the chunk size in bytes
        sample = pd.read_csv(self.data_dir + self.fname, nrows=1000)
        approx_row_size = sample.memory_usage(index=False, deep=True).sum() / len(sample)
        print(f"Approx. bytes per row: {approx_row_size}")
        target_chunk_size_bytes = chunksize*1000 * 1024**2  # 500 MB
        chunksize_rows = int(target_chunk_size_bytes / approx_row_size)
        print(f"Rows per chunk: {chunksize_rows}")

        while True:
            start = time.time()

            # Read an initial chunk
            if chunk_num == 0:
                self.df = pd.read_csv(self.data_dir + self.fname, delimiter=',', skiprows=range(1, start_row + 1),
                                      nrows=chunksize_rows, dtype=int)
            else:
                self.df = pd.read_csv(self.data_dir + self.fname, delimiter=',', skiprows=range(0, start_row),
                                      nrows=chunksize_rows, dtype=int, names=headers, header=None)
                # self.df.columns = headers

            if self.df.empty:
                break  # done

            sync = self.df.loc[
                (self.df['overflow'] == 1) & (
                        self.df['channel'] == 0)]

            if sync.empty:
                print('Warning: Possible file chunk size too small. Did not find a laser shot event. Please use a larger '
                      "chunk size if this wasn't last chunk.")
                break
            else:
                # Cut the chunk at the last 1,63 row
                cut_idx = sync.index[-1] + 1
                self.df = self.df.iloc[:cut_idx]

            rollover = self.df.loc[(self.df['overflow'] == 1) & (
                    self.df['channel'] == 63)]  # Clock rollover ("overflow", "channel" = 1,63)
            # Max count is 2^25-1=33554431

            # Create new dataframe without rollover events
            self.df1 = self.df.drop(rollover.index)  # Remove rollover events
            self.df1 = self.df1.reset_index(drop=True)  # Reset indices

            # Identify detection events ('detect') and laser pulse events ('sync')
            detect = self.df1.loc[
                (self.df1['overflow'] == 0) & (
                        self.df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
            sync = self.df1.loc[
                (self.df1['overflow'] == 1) & (
                        self.df1['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)

            # Ignore detections that precede first laser pulse event
            start_idx = sync.index[0]
            detect = detect[detect.index > start_idx]

            # Detection "times" in clock counts. Note each clock count is equivalent to 25 ps
            sync_times = sync['dtime']  # TODO: Throw away shots whose interarrival timestamps are not 70us or close to the rollover value
            detect_times = detect['dtime']

            counts = np.diff(
                sync.index) - 1  # Number of detections per pulse (subtract 1 since sync event is included in np.diff operation)
            remainder = max(0, detect.index[-1] - sync.index[-1])  # return positive remainder. If negative, there is zero remainder.
            counts = np.append(counts, remainder)  # Include last laser shot too
            sync_ref = np.repeat(sync_times,
                                 counts)  # Repeated sync time array that stores the corresponding timestamp of the laser event. Each element has a corresponding detection event.
            shots_ref = np.repeat(np.arange(len(sync)), counts)
            shots_time = shots_ref / self.PRF  # [s] Equivalent time for each shot TODO: fix PRF estimation and use sync timestamps

            # Convert detection absolute                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           timestamps to relative timestamps
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

            name, ext = os.path.splitext(self.fname_nc)
            fname_nc_iter = f"{name}_{chunk_num}{ext}"
            preprocessed_data.to_netcdf(os.path.join((self.data_dir + self.preprocessed_dir), fname_nc_iter))

            print('\nPreprocessed #{:.0f} chunk...\nTime elapsed: {:.1f} s'.format(chunk_num, time.time()-start))
            start_row += len(self.df)
            chunk_num += 1

        print('Finished preprocessing. File created.\nTotal time elapsed: {:.1f} seconds'.format(time.time() - global_start))

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

        rbins_centers = (r_binedges + 0.5*(r_binedges[1]-r_binedges[0]))[:-1]
        flux_corrected = flux_bg_sub * (rbins_centers**2)[:, np.newaxis]  # [Hz m^2] range corrected flux (after bg subtraction)

        # TODO: range correct (R**2). Maybe make bg and range corrections a separate function?

        print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

        return {
            't_binedges': t_binedges,
            'r_binedges': r_binedges,
            'flux_bg_sub': flux_bg_sub,
            'bg_flux': bg_flux,
            'flux_corrected': flux_corrected,
            'flux_raw': flux
        }

    def plot_histogram(self, histogram_results):
        # Processed data
        flux_raw = histogram_results['flux_raw']
        flux_bg_sub = histogram_results['flux_bg_sub']        # [Hz] backscatter flux
        flux_corrected = histogram_results['flux_corrected']  # [Hz m^2] range-corrected background-subtracted flux
        bg_flux = histogram_results['bg_flux']                # [Hz] background flux
        t_binedges = histogram_results['t_binedges']          # [s] temporal bin edges
        r_binedges = histogram_results['r_binedges']          # [m] range bin edges

        # Start plotting
        print('\nStarting to generate histogram plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi, figsize=(self.figsize[0], self.figsize[1]))
        ax = fig.add_subplot(111)
        if self.flux_correct:
            mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_corrected, cmap='viridis',
                             norm=LogNorm(vmin=bg_flux*((self.bg_edges[0]+self.bg_edges[1])/2)**2, vmax=flux_corrected.max()))
            fig.suptitle('CoBaLT Range-Corrected Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Range-Corrected Flux [Hz m^2]')
        else:
            mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_raw, cmap='viridis',
                             norm=LogNorm(vmin=bg_flux, vmax=flux_raw.max()))
            fig.suptitle('CoBaLT Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title('Scale {:.1f} m x {:.2f} s'.format(self.rbinsize, self.tbinsize))
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.use_ylim else ax.set_ylim([0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.use_xlim else None
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time()-start))
        if self.save_img:
            img_fname = self.generic_fname + '_hg' + '.png'
            self.img_save_path = Path(self.data_dir + self.image_dir) / img_fname
            fname = self.get_unique_filename(self.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
        plt.show()

    def plot_scatter(self, preprocessed_results):
        # Load preprocessed data
        ranges = preprocessed_results['ranges']
        shots_time = preprocessed_results['shots_time']

        # Start plotting
        print('\nStarting to generate scatter plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi, figsize=(self.figsize[0], self.figsize[1]))
        ax = fig.add_subplot(111)
        ax.scatter(shots_time, ranges / 1e3, s=self.dot_size, linewidths=0)
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.use_ylim else ax.set_ylim([0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.use_xlim else None
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title('CoBaLT Backscatter')
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time()-start))
        if self.save_img:
            img_fname = self.generic_fname + '_scatter' + '.png'
            self.img_save_path = Path(self.data_dir + self.image_dir) / img_fname
            fname = self.get_unique_filename(self.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
        plt.show()

    def get_unique_filename(self, filename):
        base, ext = os.path.splitext(filename)
        counter = 0
        filename = f"{base}_{counter}{ext}"
        while os.path.exists(filename):
            filename = f"{base}_{counter}{ext}"
            counter += 1
        return filename


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dp = DataPreprocessor(config)
    dp.preprocess()
    if dp.histogram:
        histogram_results = dp.gen_histogram(preprocessed_results)
        dp.plot_histogram(histogram_results)
    else:
        dp.plot_scatter(preprocessed_results)


if __name__ == '__main__':
    main()


# Graveyard
    # def load_data(self):
    #     self.generic_fname = Path(self.fname).stem
    #     self.fname_nc = self.generic_fname + '_preprocessed.nc'
    #     self.file_path_pkl = Path(self.data_dir + self.preprocessed_dir) / self.fname_pkl
    #     self.file_path_nc = Path(self.data_dir + self.preprocessed_dir) / self.fname_nc
    #
    #     # Search for preexisting preprocessed datafile. Else, create one.
    #     if self.file_path_nc.exists():
    #         print('\nPreprocessed datafile already exists.')
    #     else:
    #         print('\nPreprocessed datafile not found. Starting process to create it...\n')
    #         # If pickle file does not exist yet, then create it
    #         if self.file_path_pkl.exists() == False:
    #             print('No existing pickle object found.\nCreating pickle object...')
    #             start = time.time()
    #
    #             # Look at first row to check for headers.
    #             headers = ['dev', 'sec', 'usec', 'overflow', 'channel', 'dtime', '[uint32_t version of binary data]']
    #             first_row = pd.read_csv(self.data_dir + self.fname, nrows=1, header=None).iloc[0]
    #
    #             # Check if first row is all strings or digits. If digits, likely missing the headers.
    #             # Can happen when splitting/splicing data files.
    #             if all(isinstance(x, str) for x in first_row) and not all(str(x).isdigit() for x in first_row):
    #                 df = pd.read_csv(self.data_dir + self.fname, delimiter=',')
    #             else:
    #                 df = pd.read_csv(self.data_dir + self.fname, delimiter=',', header=None)
    #                 df.columns = headers
    #             outfile = open('{}/{}/{}'.format(self.data_dir, self.preprocessed_dir, self.fname_pkl), 'wb')
    #             pickle.dump(df, outfile)
    #             outfile.close()
    #             print('Finished pickling.\nTime elapsed: {:.2f} s'.format(time.time() - start))
    #
    #         # Unpickle the data to DataFrame
    #         print('Pickle file found. Loading...')
    #         infile = open('{}/{}/{}'.format(self.data_dir, self.preprocessed_dir, self.fname_pkl), 'rb')
    #         self.df = pickle.load(infile)
    #         infile.close()
    #         print('Finished loading.')