"""
Objective: Preprocess measurement data (.ARSENL files) from CoBaLT and save to netCDF (.nc) format
"""

"work flow: load data (.ARSENL) then save data (.nc)"

import time
import pandas as pd
from pathlib import Path
import os
import glob
from datetime import datetime

from utils.path_utils import find_data_path, parse_filename
from io_utils.data_reader import stack_buffer, trim_chunk, remove_rollover, locate_detect_sync
from io_utils.data_writer import write_netCDF
from instrument.timing import calibrate_time, timestamps_to_ranges

class DataPreprocessor:
    def __init__(self, config):
        # Initialize uninstantiated attributes
        self.time_delay_correct = None
        self.generic_fname = None
        self.fname_nc = None
        self.preprocess_path = None
        self.file_path_nc = None

        self.chunksize = 50_000_000  # reasonable value to produce ~700 MB size .nc files

        self.c = config['constants']['c']

        # File params
        self.date = config['file_params']['date']  # Date directory
        # Choose data directory based on OS
        self.data_dir = config['file_params']['data_dir_win'] if os.name == 'nt' else config['file_params']['data_dir_lin']
        self.fname = config['file_params']['fname']  # File name of raw data
        self.preprocessed_dir = config['file_params']['preprocessed_dir']  # Directory to store preprocessing files

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']  # [s] clock resolution

        # Process params
        self.time_delay_correct = config['process_params']['time_delay_correct']
        self.range_shift_correct = config['process_params']['range_shift_correct']
        self.range_shift = config['process_params']['range_shift']

    def write_ARSENL_netCDF(self):
        """
        Objective: Load .ARSENL file and save as netCDF (.nc) format. Data is then ready to load and visualize/process.
        """

        print('Measured data to be loaded...')
        date_str = self.date.lstrip('/')
        current = datetime.strptime(date_str, "%Y%m%d")
        threshold = datetime.strptime("2025-09-13", "%Y-%m-%d")

        # Compare
        self.time_delay_correct = False if (current < threshold) else True
        self.data_dir = find_data_path(self.data_dir)

        # Important file names and paths defined here
        self.generic_fname = Path(self.fname).stem
        self.fname_nc = self.generic_fname + '.nc'
        self.preprocess_path = self.data_dir + self.preprocessed_dir + self.date
        self.file_path_nc = Path(self.preprocess_path) / self.fname_nc
        low_gain, timestamp = parse_filename(self.fname_nc)

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
            read_path = self.data_dir + self.date + self.fname
            for chunk in pd.read_csv(read_path, delimiter=',', chunksize=self.chunksize,
                                     dtype=int, on_bad_lines='skip', encoding_errors='ignore'):
                """
                -------------------------------------------------------
                PART 1: CLEAN CHUNK BEFORE CALCULATING RANGES AND SHOTS
                -------------------------------------------------------
                """
                shot_diff = 0  # Initialize

                # Buffer contains data cutoff from end of previous chunk
                if not buffer.empty:
                    chunk = stack_buffer(buffer, chunk)

                # Empty chunk means chunks and buffer are fully processed. Finish.
                if chunk.empty:
                    break  # done

                # Retain data up to last sync-event row in chunk
                trim_result = trim_chunk(chunk)
                if (trim_result == 0) or (trim_result == 1):
                    break
                else:
                    sync, chunk_trim, buffer = trim_result

                # If this is the first chunk, then remove calibration section from data
                if chunk_iter == 0:
                    if self.time_delay_correct:
                        chunk_trim, shot_diff = calibrate_time(
                            sync=sync,
                            chunk_trim=chunk_trim,
                            low_gain=low_gain,
                            fname=self.fname,
                            data_dir=self.data_dir,
                            date=self.date,
                            chunksize=self.chunksize,
                            PRF=self.PRF
                        )

                        input(
                            "Calculating time shift between channels. User needs to ensure calibration was conducted "
                            "for this measurement. Press any key to continue..."
                        )
                    else:
                        shot_diff = 0
                        input(
                            "Will not calculate time shift between channels. User needs to ensure this is intended. "
                            "Press any key to continue..."
                        )

                # Create new dataframe with rollover events removed
                chunk_fin = remove_rollover(chunk_trim)

                # Identify detection events ('detect') and laser pulse events ('sync')
                detect, sync = locate_detect_sync(chunk_fin, chunk_iter)

                """ 
                ----------------------------------------------
                PART 2: CONVERT TIMESTAMPS TO RANGES AND SHOTS
                ----------------------------------------------
                """

                ranges, shots_time, last_sync = timestamps_to_ranges(
                    sync=sync,
                    detect=detect,
                    last_sync=last_sync,
                    unwrap_modulo=self.unwrap_modulo,
                    clock_res=self.clock_res,
                    c=self.c,
                    PRF=self.PRF,
                    range_shift_correct=self.range_shift_correct,
                    range_shift=self.range_shift,
                    low_gain=low_gain
                )

                """ 
                ---------------------------
                PART 3: WRITE OUT TO NETCDF 
                ---------------------------
                """

                # Save preprocessed data to netCDF
                write_netCDF(ranges, shots_time, shot_diff, self.fname_nc, chunk_iter, self.preprocess_path)

                chunk_iter += 1
                time_end_chunk = time.time()
                print('\nPreprocessed #{:.0f} chunk...\n'
                      'Time elapsed: {:.1f} s'.format(chunk_iter, time_end_chunk - time_update[-1]))
                time_update.append(time_end_chunk)

            print('Finished preprocessing. File created.\n'
                  'Total time elapsed: {:.1f} seconds'.format(time.time() - start))

        return self.preprocess_path, self.generic_fname, low_gain, timestamp
