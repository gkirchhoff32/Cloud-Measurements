"""
Objective: Define functions for loading data from .nc files (post preprocessing)
"""

import numpy as np
import glob
import os
import re
import xarray as xr
from pathlib import Path

from io_utils.data_reader import get_chunk_files, get_chunk_num
from utils.array_utils import flatten_ranges_shots
from utils.pipeline_utils import identify_tmin_tmax, check_overlap
from utils.path_utils import find_data_path

class netCDFLoader:
    def __init__(self, config, use_sim):
        # Instantiate attributes
        self.ranges_tot = []
        self.shots_time_tot = []

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # # File params
        self.fname = config['file_params']['fname']
        if use_sim:
            self.save_dir = config['file_params']['save_dir']
            self.func_shape = config['file_params']['func_shape']

        # System params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Process params
        self.load_xlim = config['process_params']['load_xlim']  # TRUE value limits range when generating histogram

        # Plot params
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits

    def load_chunks(self, preprocess_path, generic_fname):
        """
        When .ARSENL datasets are too large, the preprocessor method will save the necessary DataArray variables to
        netCDF file chunks. To load these variables, it's important to load chunks and store values as class properties
        for future handling.
        """
        # Only load chunks that fall within specified time window. Else load over the entire time range.
        tmin, tmax = identify_tmin_tmax(self.xlim, self.load_xlim, self.PRF)

        # Get all chunk files
        files = get_chunk_files(preprocess_path, generic_fname)

        loaded = 0
        ranges_tot = []
        shots_time_tot = []
        covered_start, covered_end = False, False
        print('Locating and loading relevant netcdf data chunks...')
        for file_path in files:
            # Open metadata only (don't load full data)
            with xr.open_dataset(file_path, decode_times=False) as ds:
                # Get number before .nc for chunk
                chunk = get_chunk_num(file_path)

                # Check for overlap. Load if it does. Skip if not.
                result, loaded, covered_start, covered_end = check_overlap(
                    ds,
                    tmin,
                    tmax,
                    chunk,
                    file_path,
                    ranges_tot,
                    shots_time_tot,
                    loaded,
                    covered_start,
                    covered_end
                )
                if result == 0:
                    break

        print('Loaded {} files'.format(loaded))

        # Flatten ranges and shots lists into 1D
        ranges, shots_time = flatten_ranges_shots(ranges_tot, shots_time_tot)

        return ranges, shots_time

    def load_sim_data(self):
        # Now load data
        home = str(Path.home())
        ds = xr.open_dataset(home + self.save_dir + self.func_shape + self.fname)
        generic_fname = Path(self.fname).stem

        cnts = ds.time_tag
        dt = ds.dt_sim

        flight_times = cnts * dt  # [s]
        ranges = flight_times * self.c / 2  # [m]
        shots_time = ds.time_tag_sync_index / self.PRF  # [s]

        return ranges.to_numpy(), shots_time.to_numpy(), generic_fname

