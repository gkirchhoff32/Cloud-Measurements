


import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from pathlib import Path
import xarray as xr
import os
import glob
import re
from datetime import datetime


class DataProcessor:
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.plotter = DataPlotter(config)
        self.deadtime = DeadtimeCorrect(config)

    def run(self):

        self.loader.preprocess()
        if self.deadtime.mueller:
            histogram_results = self.loader.gen_histogram(self.deadtime)
            self.deadtime.mueller_correct(histogram_results, self.loader, self.plotter)
        else:
            if self.plotter.histogram:
                histogram_results = self.loader.gen_histogram(self.deadtime)
                self.plotter.plot_histogram(histogram_results, self.loader)
            else:
                self.plotter.plot_scatter(self.loader)


class DeadtimeCorrect:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']  # [s] clock resolution
        self.deadtime_hg = config['system_params']['deadtime_hg']  # [s] high-gain detector deadtime
        self.deadtime_lg = config['system_params']['deadtime_lg']  # [s] low-gain detector deadtime

        self.mueller = config['process_params']['mueller']  # TRUE value applies Mueller Correction

    def mueller_correct(self, histogram_results, loader, plotter):
        flux_raw = histogram_results['flux_raw']
        r_binedges = histogram_results['r_binedges']
        deadtime = self.deadtime_lg if loader.low_gain else self.deadtime_hg

        # Calculate flux estimate based on Mueller Correction
        flux_mueller = flux_raw / (1 - deadtime * flux_raw)  # [Hz]

        # Estimate background flux
        show_hg = input("\nShow histogram to estimate background? (Y/N)")
        while True:
            if (show_hg == 'Y') or (show_hg == 'y'):
                plot_xlim, plot_ylim = plotter.plot_xlim, plotter.plot_ylim
                plotter.plot_xlim, plotter.plot_ylim = False, False
                plotter.plot_histogram(histogram_results, loader)
                plotter.plot_xlim, plotter.plot_ylim = plot_xlim, plot_ylim
                break
            elif (show_hg == 'N') or (show_hg == 'n'):
                break
            else:
                print('Please input "Y" or "N".')

        bg_ranges = input("\nEnter ranges [km] to estimate background (format: min1,max1): ")
        min_bg, max_bg = map(float, bg_ranges.split(","))  # [km]
        plotter.bg_edges = [min_bg, max_bg]
        bg_edges_idx = [np.argmin(np.abs(r_binedges - plotter.bg_edges[0] * 1e3)), np.argmin(np.abs(r_binedges - plotter.bg_edges[1] * 1e3))]

        # Subtract background for both deadtime corrected and uncorrected cases
        flux_bg = np.mean(flux_raw[bg_edges_idx[0]:bg_edges_idx[1], :])
        flux_bg_sub = flux_raw - flux_bg  # [Hz] flux with background subtracted
        flux_m_bg = np.mean(flux_mueller[bg_edges_idx[0]:bg_edges_idx[1], :])
        flux_m_bg_sub = flux_mueller - flux_m_bg  # [Hz] flux with mueller correction and background subtracted

        print('Background flux estimate: {:.2f} Hz'.format(flux_bg))
        if flux_bg >= 25e3:  # [Hz]
            print('Background estimate relatively high. Recommend to check background range values...')
            user_quit = input('Quit? (Y/N)')
            if (user_quit == 'Y') or (user_quit == 'y'):
                quit()

        histogram_results['flux_bg_sub'] = flux_bg_sub  # [Hz]
        histogram_results['flux_bg'] = flux_bg  # [Hz]
        plotter.cbar_max = flux_bg_sub.max()  # [Hz]
        plotter.cbar_min = flux_bg  # [Hz]
        plotter.flux_bg_sub = True
        plotter.plot_histogram(histogram_results, loader)

        # Mueller corrected and background subtracted
        histogram_results['flux_bg_sub'] = flux_m_bg_sub  # [Hz]
        histogram_results['flux_bg'] = flux_m_bg  # [Hz]
        plotter.plot_histogram(histogram_results, loader)

    def measure_deadtime(self, ranges, loader):
        """
        Empirically measure and plot the deadtime from time tags from saturating measurement.

        Inputs:
            ranges (nx1): [m] range array
        """

        # Calculate inter-arrival flight times
        flight_times = ranges / self.c * 2  # [s]
        intr_times = np.diff(flight_times)  # [s] Inter-arrival times

        start = time.time()
        max_val = 100e-9  # [s] capping range to explore deadtime value from inter-arrival times. Usually around 20-40
        # ns for SPCM detectors.
        bins = np.arange(0, max_val, self.clock_res)  # [s]
        cnts, bin_edges = np.histogram(intr_times, bins)  # [], [s]
        print('Time elapsed {} s'.format(time.time() - start))

        # Deadtime estimate is the inter-arrival histogram peak location
        binsize = bin_edges[1] - bin_edges[0]  # [s]
        bin_centers = bin_edges[:-1] + 0.5 * binsize  # [s]
        bin_widths = np.diff(bin_edges)  # [s]
        deadtime_estimate = bin_centers[np.argmax(cnts)]  # [s]
        print('Deadtime Estimate: {:.4f} ns'.format(deadtime_estimate * 1e9))
        setattr(self, f"deadtime_{'lg' if loader.low_gain else 'hg'}", deadtime_estimate)

        # Plot inter-arrival time histogram
        xmax = deadtime_estimate * 1e9  # [ns]
        ymax = cnts.max()
        fig = plt.figure(dpi=400, figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.bar(bin_centers * 1e9, cnts, width=bin_widths*1e9, align="edge", edgecolor="black")
        ax.annotate("Deadtime Estimate {:.1f} ns".format(deadtime_estimate*1e9), xy=(xmax, ymax),
                    xytext=(xmax, 1.3 * ymax), ha="center", va="bottom", arrowprops=dict(arrowstyle="->"))
        ax.set_xlabel("$\Delta t$ [ns]")
        ax.set_ylabel("Counts")
        ax.set_title('Inter-Arrival Times {} Channel\n{}'.format("Low-Gain" if loader.low_gain is True else "High-Gain", loader.timestamp))
        ax.set_ylim(0, 1.5 * ymax)
        plt.tight_layout()
        plt.show()


class DataPlotter:
    def __init__(self, config):
        self.flux_bg_sub = False

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Plot params
        self.chunksize = 50_000_000  # reasonable value to produce ~700 MB size .nc files
        self.plot_xlim = config['plot_params']['plot_xlim']  # TRUE value limits range only when plotting
        self.plot_ylim = config['plot_params']['plot_ylim']  # TRUE value limits range only when plotting
        self.tbinsize = config['plot_params']['tbinsize']  # [s] temporal bin size
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.bg_edges = config['plot_params']['bg_edges']  # [m] range background window
        self.dpi = config['plot_params']['dpi']  # dots-per-inch
        self.figsize = config['plot_params']['figsize']  # figure size in inches
        self.ylim = config['plot_params']['ylim']  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits
        self.histogram = config['plot_params']['histogram']  # Plot histogram if TRUE, else scatter plot
        self.save_img = config['plot_params']['save_img']  # Save images if TRUE
        self.save_dpi = config['plot_params']['save_dpi']  # DPI for saved images
        self.dot_size = config['plot_params']['dot_size']  # Dot size for 'axes.scatter' 's' param
        self.flux_correct = config['plot_params']['flux_correct']  # TRUE will plot flux that has been background
        # subtracted and range corrected
        self.chunk_start = config['plot_params']['chunk_start']  # Chunk to start plotting from
        self.chunk_num = config['plot_params']['chunk_num']  # Number of chunks to plot. If exceeds remaining chunks,
        # then it will plot the available ones
        self.alpha = config['plot_params']['alpha']  # alpha value when plotting

    def plot_histogram(self, histogram_results, loader):
        """
        Plot histogram using results from "gen_histogram" method

        Inputs:
            histogram_results: output dictionary from "gen_histogram" function
        """
        # Processed data
        flux_raw = histogram_results['flux_raw']
        flux_bg_sub = histogram_results['flux_bg_sub']
        flux_corrected = histogram_results['flux_corrected']  # [Hz m^2] range-corrected background-subtracted flux
        bg_flux = histogram_results['bg_flux']  # [Hz] background flux
        t_binedges = histogram_results['t_binedges']  # [s] temporal bin edges
        r_binedges = histogram_results['r_binedges']  # [m] range bin edges

        # Start plotting
        print('\nStarting to generate histogram plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi, figsize=(self.figsize[0], self.figsize[1]))
        ax = fig.add_subplot(111)
        if self.flux_correct:
            mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_corrected, cmap='viridis',
                                 norm=LogNorm(vmin=bg_flux * ((self.bg_edges[0] + self.bg_edges[1]) / 2) ** 2,
                                              vmax=flux_corrected.max()))
            fig.suptitle('CoBaLT Range-Corrected Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Range-Corrected Flux [Hz m^2]')
        elif self.flux_bg_sub:
            mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_bg_sub, cmap='viridis',
                                 norm=LogNorm(vmin=self.cbar_min, vmax=self.cbar_max))
            fig.suptitle('CoBaLT Background-Subtracted Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
        else:
            mesh = ax.pcolormesh(t_binedges, r_binedges / 1e3, flux_raw, cmap='viridis',
                                 norm=LogNorm(vmin=flux_raw[flux_raw > 0].min(), vmax=flux_raw.max()))
            fig.suptitle('CoBaLT Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title('Scale {:.1f} m x {:.2f} s\n{} {}'.format(self.rbinsize, self.tbinsize,
                                                               "Low Gain" if loader.low_gain else "High Gain",
                                                               loader.timestamp))
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.plot_ylim else ax.set_ylim(
            [0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.plot_xlim else None
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        if self.save_img:
            img_fname = loader.generic_fname + '_hg' + '.png'
            loader.img_save_path = Path(loader.data_dir + loader.image_dir + loader.date) / img_fname
            fname = loader.get_unique_filename(loader.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
        plt.show()

    def plot_scatter(self, loader):
        """
        Display scatter plot of time tags
        """

        loader.load_chunk()

        ranges = np.concatenate([da.values.ravel() for da in loader.ranges_tot])
        shots_time = np.concatenate([da.values.ravel() for da in loader.shots_time_tot])

        # Start plotting
        print('\nStarting to generate scatter plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi, figsize=(self.figsize[0], self.figsize[1]))
        ax = fig.add_subplot(111)
        ax.scatter(shots_time, ranges / 1e3, s=self.dot_size, alpha=self.alpha, linewidths=0)
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.plot_ylim else ax.set_ylim(
            [0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.plot_xlim else None
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title(
            'CoBaLT Backscatter\n{} {}'.format("Low Gain" if loader.low_gain else "High Gain", loader.timestamp))
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        if self.save_img:
            print('Starting to save image...')
            start = time.time()
            img_fname = loader.generic_fname + '_scatter' + '.png'
            loader.img_save_path = Path(loader.data_dir + loader.image_dir + loader.date) / img_fname
            fname = loader.get_unique_filename(loader.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
            print('Finished saving plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        plt.show()


class DataLoader:
    def __init__(self, config):
        self.generic_fname = None
        self.fname_nc = None
        self.preprocess_path = None
        self.ranges_tot = []
        self.shots_time_tot = []
        self.chunksize = 50_000_000  # reasonable value to produce ~700 MB size .nc files

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']  # [s] clock resolution

        # Process params
        self.load_ylim = config['process_params']['load_ylim']  # TRUE value limits range when generating histogram
        self.load_xlim = config['process_params']['load_xlim']  # TRUE value limits range when generating histogram
        self.time_delay_correct = config['process_params']['time_delay_correct']
        self.range_shift_correct = config['process_params']['range_shift_correct']
        self.range_shift = config['process_params']['range_shift']
        self.estimate_deadtime = config['process_params']['estimate_deadtime']  # TRUE value estimates and uses empirical deadtime

        # File params
        self.fname = config['file_params']['fname']  # File name of raw data
        self.data_dir = config['file_params']['data_dir']  # Directory of raw data
        self.preprocessed_dir = config['file_params']['preprocessed_dir']  # Directory to store preprocessing files
        self.image_dir = config['file_params']['image_dir']  # Directory to save images
        self.date = config['file_params']['date']  # YYYYMMDD: points to identically named directories

        # Plot params
        self.ylim = config['plot_params']['ylim']  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits
        self.tbinsize = config['plot_params']['tbinsize']  # [s] temporal bin size
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.chunk_start = config['plot_params']['chunk_start']  # Chunk to start plotting from
        self.chunk_num = config['plot_params']['chunk_num']  # Number of chunks to plot. If exceeds remaining chunks,
        # then it will plot the available ones
        self.bg_edges = config['plot_params']['bg_edges']  # [m] range background window

    def preprocess(self):
        """
        Perform simple preprocessing. For example, pull out shot counts and measured ranges from the binary file format.
        Assign important variables to xarray Dataset, e.g., "ranges" and "shots time".
        Then convert ".ARSENL" data to netCDF format (".nc") and write out file.
        """

        # Important file names and paths
        self.generic_fname = Path(self.fname).stem
        self.fname_nc = self.generic_fname + '.nc'
        self.preprocess_path = self.data_dir + self.preprocessed_dir + self.date
        self.file_path_nc = Path(self.preprocess_path) / self.fname_nc
        self.parse_filename()

        # Load preprocessed data (chunk) if exists. Otherwise, preprocess and save out results to .nc file.
        if glob.glob(os.path.join(self.preprocess_path, self.generic_fname + '_*.nc')):
            print('\nPreprocessed data file(s) found. No need to create new one(s)...')
        else:
            print('\nPreprocessed data file(s) not found. Creating file(s)...\nStarting preprocessing...')
            start = time.time()
            time_update = [start]  # List to store elapsed times after each chunk is processed

            # Load chunks one at a time and calculate measurements from file format
            chunk_iter = 0
            last_sync = -1  # Track the last shot time per chunk
            buffer = pd.DataFrame()  # store leftover rows across chunks
            for chunk in pd.read_csv(self.data_dir + self.date + self.fname, delimiter=',', chunksize=self.chunksize,
                                     dtype=int, on_bad_lines='skip', encoding_errors='ignore'):

                """
                -------------------------------------------------------
                PART 1: CLEAN CHUNK BEFORE CALCULATING RANGES AND SHOTS
                -------------------------------------------------------
                """

                # Buffer contains data cutoff from end of previous chunk
                if not buffer.empty:
                    chunk = pd.concat([buffer, chunk], ignore_index=True)

                # Empty chunk means chunks and buffer are fully processed. Finish.
                if chunk.empty:
                    break  # done

                # Retain data up to last sync-event row in chunk
                sync = chunk.loc[(chunk['overflow'] == 1) & (chunk['channel'] == 0)]
                if sync.empty:
                    print('Warning: Possible file chunk size too small. Did not find a laser shot event. Please use a '
                          "larger chunk size if this wasn't the last chunk.")
                    break
                elif len(sync) == 1:
                    # If the sync length is only one, then reached the last laser shot. Finish.
                    print('No more chunks to process.')
                    break
                else:
                    # Cut the chunk at the last 1,0 (sync event) row
                    cut_idx = sync.index[-1]
                    chunk_trim = chunk.iloc[:cut_idx]
                    buffer = chunk.iloc[cut_idx:]

                # If this is the first chunk, then remove calibration section from data
                if chunk_iter == 0:
                    if self.time_delay_correct is True:
                        chunk_trim, shot_diff = self.calibrate_time(sync, chunk_trim)
                        input("Calculating time shift between channels. User needs to ensure calibration was conducted "
                              "for this measurement. Press any key to continue...")
                    else:
                        shot_diff = 0
                        input("Will not calculate time shift between channels. User needs to ensure this is intended. "
                              "Press any key to continue...")

                # Clock rollover ("overflow", "channel" = 1,63). Max count is 2^25-1=33554431
                rollover = chunk_trim.loc[(chunk_trim['overflow'] == 1) & (chunk_trim['channel'] == 63)]

                # Create new dataframe without rollover events
                chunk_fin = chunk_trim.drop(rollover.index)  # Remove rollover events
                chunk_fin = chunk_fin.reset_index(drop=True)  # Reset indices

                # Identify detection events ('detect') and laser pulse events ('sync')
                detect = chunk_fin.loc[
                    (chunk_fin['overflow'] == 0) & (
                            chunk_fin['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
                sync = chunk_fin.loc[
                    (chunk_fin['overflow'] == 1) & (
                            chunk_fin['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)

                # Ignore detections that precede first laser pulse event
                if chunk_iter == 0:
                    start_idx = sync.index[0]
                    detect = detect[detect.index > start_idx]

                """ 
                ----------------------------------------------
                PART 2: CONVERT TIMESTAMPS TO RANGES AND SHOTS
                ----------------------------------------------
                """

                # Detection "times" in clock counts. Note each clock count is equivalent to 25 ps
                sync_times = sync['dtime']
                # TODO: Throw away shots whose interarrival timestamps are not 70us or close to the rollover value
                detect_times = detect['dtime']

                counts = np.diff(sync.index) - 1  # Number of detections per pulse (subtract 1 since sync event is
                # included in 'np.diff' operation)
                remainder = max(0, detect.index[-1] - sync.index[-1])  # return positive remainder. If negative, there
                # is zero remainder.
                counts = np.append(counts, remainder)  # Include last laser shot too
                sync_ref = np.repeat(sync_times, counts)  # Repeated sync time array that stores the corresponding
                # timestamp of the laser event. Each element has a corresponding detection event.
                shots_ref = np.repeat(np.arange(start=last_sync + 1, stop=(last_sync + 1) + len(sync)), counts)
                last_sync = shots_ref[-1]  # Track last sync event
                shots_time = shots_ref / self.PRF  # [s] Equivalent time for each shot TODO: fix PRF estimation and use sync timestamps

                # Convert detection times relative to most recent laser shot timestamps to relative timestamps
                detect_times_rel = detect_times.to_numpy() - sync_ref.to_numpy()

                # Handle rollover events. Add the clock rollover value to any negative timestamps.
                # A rollover is where the timestamps cycle back to 1 after the clock has reached 2^25-1.
                # This is because if detections occurred between a rollover and sync event, then corresponding
                # "detect_time_rel" element will be negative.
                rollover_idx = np.where(detect_times_rel < 0)[0]
                detect_times_rel[rollover_idx] += self.unwrap_modulo

                # Convert to flight times and range
                flight_times = detect_times_rel * self.clock_res  # [s] counts were in 25-ps increments
                ranges = flight_times * self.c / 2  # [m]

                # Range correct for path-length difference
                if (self.range_shift_correct is True) and (self.low_gain is False):
                    ranges += self.range_shift  # [m]

                """ 
                ---------------------------
                PART 3: WRITE OUT TO NETCDF 
                ---------------------------
                """

                # Save preprocessed data to netCDF
                preprocessed_data = xr.Dataset(
                    data_vars=dict(
                        ranges=ranges,
                        shots_time=shots_time,
                        shot_diff=shot_diff
                    )
                )

                name, ext = os.path.splitext(self.fname_nc)
                fname_nc_iter = f"{name}_{chunk_iter}{ext}"
                preprocessed_data.to_netcdf(os.path.join((self.data_dir + self.preprocessed_dir + self.date),
                                                         fname_nc_iter))

                chunk_iter += 1
                time_end_chunk = time.time()
                print('\nPreprocessed #{:.0f} chunk...\n'
                      'Time elapsed: {:.1f} s'.format(chunk_iter, time_end_chunk - time_update[-1]))
                time_update.append(time_end_chunk)

            print('Finished preprocessing. File created.\n'
                  'Total time elapsed: {:.1f} seconds'.format(time.time() - start))

    def gen_histogram(self, deadtime):
        """
        Calculate histogram from range and shots properties of netCDF file. Flux is calculated counts per range bin per
        integrated shots.

        Returns:
            t_binedges [s]: ([n+1]x1) histogram time bin-edge values
            r_binedges [m]: ([m+1]x1) histogram range bin-edge values
            flux_bg_sub [Hz]: (nxm) histogram flux values
            bg_flux [Hz]: (float) background flux estimate
            flux_corrected [Hz m^2]: (nxm) range- and background-corrected flux
            flux_raw [Hz]: (nxm) uncorrected flux
        """

        if self.load_xlim:
            min_time = self.xlim[0]  # [s]
            max_time = self.xlim[1]  # [s]
        if self.load_ylim:
            min_range = self.ylim[0] * 1e3  # [m]
            max_range = self.ylim[1] * 1e3  # [m]

        self.load_chunk()

        ranges = np.concatenate([da.values.ravel() for da in self.ranges_tot])
        shots_time = np.concatenate([da.values.ravel() for da in self.shots_time_tot])

        if self.estimate_deadtime:
            deadtime.measure_deadtime(ranges, self)

        if self.load_xlim:
            max_shots_idx = np.argmin(np.abs(shots_time - max_time))
            min_shots_idx = np.argmin(np.abs(shots_time - min_time))
            shots_time = shots_time[min_shots_idx:max_shots_idx]
            ranges = ranges[min_shots_idx:max_shots_idx]

        print('\nStarting to generate histogram...')
        start = time.time()
        tbins = np.arange(shots_time[0], shots_time[-1], self.tbinsize)  # [s]
        if self.load_ylim:
            rbins = np.arange(min_range, max_range, self.rbinsize)  # [m]
        else:
            rbins = np.arange(0, self.c / 2 / self.PRF, self.rbinsize)  # [m]

        H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
        H = H.T  # flip axes
        flux = H / (self.rbinsize / self.c * 2) / (
                self.tbinsize * self.PRF)  # [Hz] Backscatter flux $\Phi = n/N/(\Delta t)$,

        # Estimate background flux
        bg_edges_idx = [np.argmin(np.abs(rbins - self.bg_edges[0])), np.argmin(np.abs(rbins - self.bg_edges[1]))]
        bg_flux = np.mean(flux[bg_edges_idx[0]:bg_edges_idx[1], :])
        flux_bg_sub = flux - bg_flux  # [Hz] flux with background subtracted

        rbins_centers = (r_binedges + 0.5 * (r_binedges[1] - r_binedges[0]))[:-1]
        flux_corrected = flux_bg_sub * (rbins_centers ** 2)[:, np.newaxis]  # [Hz m^2] range corrected flux (after bg
        # subtraction)

        print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

        return {
            't_binedges': t_binedges,
            'r_binedges': r_binedges,
            'flux_bg_sub': flux_bg_sub,
            'bg_flux': bg_flux,
            'flux_corrected': flux_corrected,
            'flux_raw': flux
        }

    def calibrate_time(self, sync, chunk_trim):
        """
        Locates calibration section of measurement (i.e., beginning of acquisition where the laser is turned
        off then on again). Removes this section and, if this is the low-gain data, loads the high-gain data
        too to calculate the relative temporal shift between channels.

        Params:
            sync: DataFrame rows of sync events from datafile
            chunk_trim: Output from trim chunk function, which is trims dataset up to last laser shot event
        """

        print('Starting to calculate calibration time shift...')

        # Calibration segment is determined by detecting large gap in sync pulse detections.
        sync_idx = sync.index
        gaps = sync_idx.to_series().diff().fillna(1)
        gap_start_idx = gaps.index[gaps.values.argmax() - 1]  # last laser shot before beginning of cal gap
        cal_sync = sync.loc[:gap_start_idx]  # cut everything past the beginning calibration section
        cal_shot_num = len(cal_sync)

        # If this is low-gain data, measure time delay using calibration section from high-gain
        # data too
        if self.low_gain:
            # Load small first chunk of high-gain file too
            hg_fname = self.fname.replace("Dev_1", "Dev_0")  # high-gain filename
            chunk_hg = next(pd.read_csv(self.data_dir + self.date + hg_fname, delimiter=',',
                                        chunksize=self.chunksize, dtype=int, on_bad_lines='skip',
                                        encoding_errors='ignore'))
            sync_hg = chunk_hg.loc[(chunk_hg['overflow'] == 1) & (chunk_hg['channel'] == 0)]
            sync_hg_idx = sync_hg.index
            gaps_hg = sync_hg_idx.to_series().diff().fillna(1)
            gap_start_idx_hg = gaps_hg.index[gaps_hg.values.argmax() - 1]
            cal_sync_hg = sync_hg.loc[:gap_start_idx_hg]  # cut everything past the beginning calibration
            # section
            cal_shot_num_hg = len(cal_sync_hg)

            shot_diff = cal_shot_num - cal_shot_num_hg
            time_diff = shot_diff / self.PRF  # [s]
            print('Time diff between channels: {} s'.format(time_diff))
        else:
            shot_diff = None

        gap_end_idx = gaps.idxmax()  # first laser shot after end of calibration gap
        chunk_trim = chunk_trim.loc[gap_end_idx:]

        return chunk_trim, shot_diff

    def load_chunk(self):
        """
        When .ARSENL datasets are too large, the preprocessor method will save the necessary DataArray variables to
        netCDF file chunks. To load these variables, it's important to load chunks and store values as class properties
        for future handling.
        """

        chunk = self.chunk_start
        for i in range(self.chunk_num):
            file_path = os.path.join(self.preprocess_path, self.generic_fname + f'_{chunk}.nc')
            # If file exists, load it.
            if glob.glob(file_path):
                print(f'\nPreprocessed file found: chunk #{chunk}')
                ds = xr.open_dataset(file_path)
                self.ranges_tot.append(ds['ranges'])
                self.shots_time_tot.append(ds['shots_time'])
                print('\nFile loaded.')
                chunk += 1
            # If file does not exist, either config settings for loading are wrong or ran out of chunks to load.
            else:
                print(f'\nPreprocessed file not found... check this')
                if chunk == self.chunk_start:
                    print('Starting chunk unavailable! Pick a different one.')
                    quit()
                else:
                    print('\nNo more chunks. Starting to plot...')
                    break

    def parse_filename(self):
        """
        Use standard naming convention from .ARSENL binary file to extract board number, time, and date.
        e.g., Dev_0_-_2025-09-13_21.47.46.ARSENL
        """

        match = re.match(r"Dev_(\d)_-_(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2}\.\d{2}).nc", self.fname_nc)
        if not match:
            raise ValueError(f"Filename format not recognized: {self.fname_nc}")

        # Pull out board number, date, and time.
        dev, date_str, time_str = match.groups()
        self.low_gain = True if dev == "1" else False

        # Convert to datetime if useful
        self.timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H.%M.%S")
        print("Measurement time: {}".format(self.timestamp))

    @staticmethod
    def get_unique_filename(filename):
        """
        When saving data, if filename exists, then save to file name based on iterator

        Params:
            filename (str): Original save file name
        Returns:
            filename (str): New save file name
        """

        base, ext = os.path.splitext(filename)
        counter = 0
        filename = f"{base}_{counter}{ext}"
        while os.path.exists(filename):
            filename = f"{base}_{counter}{ext}"
            counter += 1
        return filename















