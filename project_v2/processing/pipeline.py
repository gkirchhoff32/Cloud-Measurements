from io_utils.path_resolver import find_data_path
from physics.calibration import calibrate_time
from physics.timing import timestamps_to_ranges
from processing.histogram import generate_histogram

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import glob

class Pipeline:
    def __init__(self, config):
        self.config = config

        # Extract constants
        self.c = config['constants']['c']
        self.PRF = config['system_params']['PRF']
        self.unwrap_modulo = config['system_params']['unwrap_modulo']
        self.clock_res = config['system_params']['clock_res']
        self.range_shift = config['process_params']['range_shift']
        self.range_shift_correct = config['process_params']['range_shift_correct']

        # File params
        self.date = config['file_params']['date']
        self.fname = config['file_params']['fname']
        self.data_dir = find_data_path(config['file_params']['data_dir_win'])

    def read_raw_chunks(self, chunksize):
        file_path = Path(self.data_dir) / self.date / self.fname
        return pd.read_csv(
            file_path,
            delimiter=',',
            chunksize=chunksize,
            dtype=int,
            on_bad_lines='skip',
            encoding_errors='ignore'
        )

    def preprocess_chunk(self, chunk, last_sync):
        # Identify sync/detect events
        sync = chunk[(chunk['overflow'] == 1) & (chunk['channel'] == 0)]
        detect = chunk[(chunk['overflow'] == 0) & (chunk['channel'] == 0)]

        # Convert timestamps → ranges
        ranges, shots_time, last_sync = timestamps_to_ranges(
            sync_times=sync['dtime'].to_numpy(),
            detect_times=detect['dtime'].to_numpy(),
            sync_indices=sync.index.to_numpy(),
            detect_indices=detect.index.to_numpy(),
            last_sync=last_sync,
            unwrap_modulo=self.unwrap_modulo,
            clock_res=self.clock_res,
            c=self.c,
            PRF=self.PRF,
            range_shift_correct=self.range_shift_correct,
            range_shift=self.range_shift,
            low_gain=False  # placeholder for now
        )

        return ranges, shots_time, last_sync

    def write_nc_chunk(self, ranges, shots_time, chunk_idx):
        ds = xr.Dataset(
            data_vars=dict(
                ranges=('ranges', ranges),
                shots_time=('shots_time', shots_time)
            )
        )

        out_dir = Path(self.data_dir) / self.config['file_params']['preprocessed_dir'] / self.date
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{Path(self.fname).stem}_{chunk_idx}.nc"
        ds.to_netcdf(out_dir / fname)

    def load_nc_chunks(self):
        out_dir = Path(self.data_dir) / self.config['file_params']['preprocessed_dir'] / self.date
        files = sorted(glob.glob(str(out_dir / "*.nc")))

        ranges_tot = []
        shots_time_tot = []

        for f in files:
            ds = xr.open_dataset(f)
            ranges_tot.append(ds['ranges'].values)
            shots_time_tot.append(ds['shots_time'].values)

        return np.concatenate(ranges_tot), np.concatenate(shots_time_tot)

    def histogram(self, ranges, shots_time):
        return generate_histogram(
            ranges=ranges,
            shots_time=shots_time,
            rbinsize=self.config['plot_params']['rbinsize'],
            tbinsize=self.config['plot_params']['tbinsize'],
            PRF=self.PRF,
            c=self.c,
            load_xlim=self.config['process_params']['load_xlim'],
            load_ylim=self.config['process_params']['load_ylim'],
            xlim=self.config['plot_params']['xlim'],
            ylim=self.config['plot_params']['ylim'],
            active_fraction=self.config['process_params']['active_fraction'],
            deadtime=self.config['system_params']['deadtime_lg'],
            gen_hist_bg=self.config['process_params']['bg_sub']
        )

    def run_preprocessing(self):
        chunksize = self.config['file_params']['chunksize']
        last_sync = -1
        chunk_idx = 0

        for chunk in self.read_raw_chunks(chunksize):
            # Identify sync events
            sync = chunk[(chunk['overflow'] == 1) & (chunk['channel'] == 0)]
            if sync.empty:
                continue

            # Trim chunk at last sync event
            cut_idx = sync.index[-1]
            chunk_trim = chunk.iloc[:cut_idx]

            # Calibration only on first chunk
            if chunk_idx == 0:
                chunk_trim, shot_diff = calibrate_time(
                    sync=sync,
                    chunk_trim=chunk_trim,
                    low_gain=False,  # placeholder
                    fname=self.fname,
                    data_dir=self.data_dir,
                    date=self.date,
                    chunksize=chunksize,
                    PRF=self.PRF
                )

            # Convert timestamps → ranges
            ranges, shots_time, last_sync = self.preprocess_chunk(chunk_trim, last_sync)

            # Write NC chunk
            self.write_nc_chunk(ranges, shots_time, chunk_idx)

            chunk_idx += 1

        print(f"Finished preprocessing {chunk_idx} chunks.")

    def run_histogram(self):
        ranges, shots_time = self.load_nc_chunks()
        return self.histogram(ranges, shots_time)

    def run(self):
        print("Starting preprocessing...")
        self.run_preprocessing()

        print("Generating histogram...")
        hist = self.run_histogram()

        return hist

