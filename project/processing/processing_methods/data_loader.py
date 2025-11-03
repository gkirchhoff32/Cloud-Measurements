import numpy as np
import time
import pandas as pd
from pathlib import Path
import xarray as xr
import os
import glob
import re
from datetime import datetime

# TODO: Automatically detect relevant chunk to load
# TODO: Throw away shots whose interarrival timestamps are not 70us or close to the rollover value
# TODO: fix PRF estimation and use sync timestamps


class DataLoader:
    def __init__(self, config):
        self.generic_fname = None
        self.fname_nc = None
        self.preprocess_path = None
        self.deadtime = None
        self.gen_hist_bg = False
        self.ranges_tot = []
        self.shots_time_tot = []
        self.chunksize = 50_000_000  # reasonable value to produce ~700 MB size .nc files

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']  # [s] clock resolution
        self.pulse_width = config['system_params']['pulse_width']  # [s] FWHM
        self.deadtime_hg = config['system_params']['deadtime_hg']  # [s]
        self.deadtime_lg = config['system_params']['deadtime_lg']  # [s]

        # Process params
        self.load_ylim = config['process_params']['load_ylim']  # TRUE value limits range when generating histogram
        self.load_xlim = config['process_params']['load_xlim']  # TRUE value limits range when generating histogram
        self.time_delay_correct = config['process_params']['time_delay_correct']
        self.range_shift_correct = config['process_params']['range_shift_correct']
        self.range_shift = config['process_params']['range_shift']
        self.active_fraction = config['process_params']['active_fraction']
        self.fractional_bin = config['process_params']['fractional_bin']
        self.bg_sub = config['process_params']['bg_sub']  # TRUE to background calculate and subtract. FALSE to skip.

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

    def preprocess(self):
        """
        Perform simple preprocessing. For example, pull out shot counts and measured ranges from the binary file format.
        Assign important variables to xarray Dataset, e.g., "ranges" and "shots time".
        Then convert ".ARSENL" data to netCDF format (".nc") and write out file.
        """

        self.data_dir = self.find_data_path(self.data_dir)

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
                shots_time = shots_ref / self.PRF  # [s] Equivalent time for each shot

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

                # Remove invalid range values
                r_valid_idx = np.where(ranges <= (self.c / 2 / self.PRF))
                ranges = ranges[r_valid_idx]
                shots_time = shots_time[r_valid_idx]

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

    def gen_histogram(self):
        """
        Calculate histogram from range and shots properties of netCDF file. Flux is calculated counts per range bin per
        integrated shots.

        Returns:
            t_binedges [s]: ([n+1]x1) histogram time bin-edge values
            r_binedges [m]: ([m+1]x1) histogram range bin-edge values
            flux_bg_sub [Hz]: (nxm) histogram flux values
            bg_flux [Hz]: (float) background flux estimate
            flux_raw [Hz]: (nxm) uncorrected flux
        """

        self.load_chunk()

        # Unpack ranges and shots
        ranges = np.concatenate([da.values.ravel() for da in self.ranges_tot])
        shots_time = np.concatenate([da.values.ravel() for da in self.shots_time_tot])

        self.deadtime = self.deadtime_lg if self.low_gain else self.deadtime_hg

        dr_af = self.rbinsize  # [m]
        dt_af = 1 / self.PRF  # [s]

        # Round time histogram bin size that factorizes the fine-res bin size
        t_factor = max(1, round(self.tbinsize / dt_af))
        tbinsize_close = t_factor * dt_af  # [s]
        rbinsize = dr_af  # [m]

        # Set time and range windows
        deadtime_range = self.deadtime * self.c / 2  # [m]
        if self.load_xlim:
            min_time, max_time = self.xlim[0], self.xlim[1]  # [s]
        else:
            min_time, max_time = shots_time[0], shots_time[-1]  # [s]
        if self.load_ylim:
            reduce_min = deadtime_range if self.active_fraction and (deadtime_range >= rbinsize) else 0
            min_range, max_range = (self.ylim[0] * 1e3 - reduce_min), (self.ylim[1] * 1e3)  # [m]
        else:
            reduce_min = 0
            min_range, max_range = 0, (self.c / 2 / self.PRF)  # [m]

        if self.load_xlim:
            max_shots_idx = np.argmin(np.abs(shots_time - max_time))
            min_shots_idx = np.argmin(np.abs(shots_time - min_time))
            shots_time = shots_time[min_shots_idx:max_shots_idx]
            ranges = ranges[min_shots_idx:max_shots_idx]

        if self.gen_hist_bg:
            print('Using approximate resolutions for background estimate: {:.3e} m x {:.3e} s.'.format(rbinsize, tbinsize_close))
        else:
            print('Using resolutions: {:.3e} m x {:.3e} s.'.format(rbinsize, tbinsize_close))

        self.rbinsize = rbinsize
        self.tbinsize = tbinsize_close
        print('Actual range and time bin sizes: {:.3e} m x {:.3e} s'.format(self.rbinsize, self.tbinsize))

        if self.gen_hist_bg:
            print('\nStarting to generate histogram for background estimate...')
        else:
            print('\nStarting to generate histogram...')

        start = time.time()
        tbins = np.arange(shots_time[0], shots_time[-1], self.tbinsize)  # [s]
        if self.load_ylim:
            rbins = np.arange(min_range, max_range + self.rbinsize, self.rbinsize)  # [m]
        else:
            rbins = np.arange(0, self.c / 2 / self.PRF + self.rbinsize, self.rbinsize)  # [m]

        # Generate histogram
        H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
        H = H.T  # flip axes
        flux = H / (self.rbinsize / self.c * 2) / (self.tbinsize * self.PRF)  # [Hz] Backscatter flux

        print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

        return {
            't_binedges': t_binedges,
            'r_binedges': r_binedges,
            'flux_raw': flux,
            'cnts_raw': H,
            'reduce_min': reduce_min
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
            chunk_hg = next(pd.read_csv(self.data_dir + self.date + hg_fname,
                                        delimiter=',',
                                        chunksize=self.chunksize,
                                        dtype=int,
                                        on_bad_lines='skip',
                                        encoding_errors='ignore')
                            )
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
        tmin = self.xlim[0]  # [s]
        tmax = self.xlim[1]  # [s]

        # Set tmin to first shot time if minimum value is 0
        tmin = 1 / self.PRF if tmin == 0 else tmin

        # chunk = self.chunk_start
        loaded = 0
        covered_start = False
        covered_end = False

        # Get all chunk files
        files = glob.glob(os.path.join(self.preprocess_path, f'{self.generic_fname}_*.nc'))

        # --- 🔧 Sort files numerically by their chunk index ---
        def extract_chunk_number(path):
            match = re.search(r'_(\d+)\.nc$', os.path.basename(path))
            return int(match.group(1)) if match else -1

        files = sorted(files, key=extract_chunk_number)

        # files = sorted(glob.glob(os.path.join(self.preprocess_path, f'{self.generic_fname}_*.nc')))
        print('Locating and loading relevant netcdf data chunks...')
        for file_path in files:

            # Open metadata only (don't load full data)
            with xr.open_dataset(file_path, decode_times=False) as ds:
                # Get the filename
                fname = os.path.basename(file_path)

                # Extract the final number before .nc
                match = re.search(r'_(\d+)\.nc$', fname)
                if match:
                    chunk = int(match.group(1))

                # Get the first and last time values (assume shots_time is 1D and sorted)
                t0 = ds['shots_time'].isel(shots_time=0).item()
                t1 = ds['shots_time'].isel(shots_time=-1).item()

                overlaps = (t1 >= tmin) and (t0 <= tmax)
                # Check for overlap
                if overlaps:
                    print(f'\nIncluding chunk #{chunk}: {t0:.2f}–{t1:.2f}s overlaps {tmin:.2f}–{tmax:.2f}s')
                    # Load the full dataset only now
                    ds_full = xr.open_dataset(file_path)
                    self.ranges_tot.append(ds_full['ranges'])
                    self.shots_time_tot.append(ds_full['shots_time'])
                    print('File loaded.')
                    loaded += 1

                    # Update coverage flags
                    if t0 <= tmin:
                        covered_start = True
                    if t1 >= tmax:
                        covered_end = True

                    # Stop early if full range is covered
                    if covered_start and covered_end:
                        print(f'\n✅ Full time range {tmin:.2f}–{tmax:.2f}s covered by loaded chunks.')
                        break
                else:
                    print(f'Skipping chunk #{chunk}: {t0:.2f}–{t1:.2f}s not in range {tmin:.2f}–{tmax:.2f}s')
        print('Loaded {} files'.format(loaded))

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

    def calc_bg(self, flux, rbins, tbins, bg_r_edges, bg_t_edges):
        # Estimate background flux
        if self.bg_sub:
            bg_r_edges_idx = [np.argmin(np.abs(rbins - bg_r_edges[0])), np.argmin(np.abs(rbins - bg_r_edges[1]))]
            bg_t_edges_idx = [np.argmin(np.abs(tbins - bg_t_edges[0])), np.argmin(np.abs(tbins - bg_t_edges[1]))]
            bg_flux = np.mean(flux[bg_r_edges_idx[0]:bg_r_edges_idx[1], bg_t_edges_idx[0]:bg_t_edges_idx[1]])  # [Hz]
        else:
            bg_flux = 0  # [Hz]

        return bg_flux

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

    @staticmethod
    def find_data_path(data_dir):
        """
        Automatically detect relevant file path for data loading. Update "candidate_roots" list if new data file paths
        are required.
        """
        target_subdir = data_dir

        # Likely parent roots — you can add others if needed
        candidate_roots = [
            Path("F:/"),
            Path("C:/Users/Grant"),
        ]

        for root in candidate_roots:
            candidate = root / target_subdir
            if candidate.exists():
                print(f"Detected data directory: {candidate}")
                return str(candidate)

        raise FileNotFoundError("Could not locate the 'OneDrive - UCB-O365' data directory.")

