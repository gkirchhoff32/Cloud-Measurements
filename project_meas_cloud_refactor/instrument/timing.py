"""
Objective: Calibrate any timing discrepancies associated with CoBaLT.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def timestamps_to_ranges(
        sync,
        detect,
        last_sync,
        unwrap_modulo,
        clock_res,
        c,
        PRF,
        range_shift_correct,
        range_shift,
        low_gain
):
    """
    Convert raw timestamps into ranges and shot times.

    Returns:
        ranges (np.ndarray)
        shots_time (np.ndarray)
        last_sync (int)  # updated
    """

    # Detection "times" in clock counts
    sync_times = sync['dtime']
    detect_times = detect['dtime']

    counts = np.diff(sync.index) - 1

    remainder = max(0, detect.index[-1] - sync.index[-1])  # return positive remainder. If negative, there
    # is zero remainder.
    counts = np.append(counts, remainder)  # Include last laser shot too
    sync_ref = np.repeat(sync_times, counts)  # Repeated sync time array that stores the corresponding
    # timestamp of the laser event. Each element has a corresponding detection event.
    shots_ref = np.repeat(np.arange(start=last_sync + 1, stop=(last_sync + 1) + len(sync)), counts)
    last_sync = shots_ref[-1]  # Track last sync event
    shots_time = shots_ref / PRF  # [s] Equivalent time for each shot

    detect_times_rel = detect_times.to_numpy() - sync_ref.to_numpy()

    # Handle rollover events. Add the clock rollover value to any negative timestamps.
    # A rollover is where the timestamps cycle back to 1 after the clock has reached 2^25-1.
    # This is because if detections occurred between a rollover and sync event, then corresponding
    # "detect_time_rel" element will be negative.
    rollover_idx = np.where(detect_times_rel < 0)[0]
    detect_times_rel[rollover_idx] += unwrap_modulo

    flight_times = detect_times_rel * clock_res
    ranges = flight_times * c / 2

    # Remove invalid range values
    r_valid_idx = np.where(ranges <= (c / 2 / PRF))
    ranges = ranges[r_valid_idx]
    shots_time = shots_time[r_valid_idx]

    if (range_shift_correct is True) and (low_gain is False):
        ranges += range_shift

    return ranges, shots_time, last_sync

def calibrate_time(sync, chunk_trim, low_gain, fname, data_dir, date, chunksize, PRF):
    """
    Calibrate the relative time shift between gain channels.
    Returns:
        chunk_trim (DataFrame)
        shot_diff (float or None)
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
    if low_gain:
        # Load small first chunk of high-gain file too
        hg_fname = fname.replace("Dev_1", "Dev_0")  # high-gain filename
        chunk_hg = next(pd.read_csv(data_dir + date + hg_fname,
                                    delimiter=',',
                                    chunksize=chunksize,
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
        time_diff = shot_diff / PRF  # [s]
        print('Time diff between channels: {} s'.format(time_diff))
    else:
        shot_diff = None

    gap_end_idx = gaps.idxmax()  # first laser shot after end of calibration gap
    chunk_trim = chunk_trim.loc[gap_end_idx:]

    return chunk_trim, shot_diff

def adjust_daylight_savings(date_str, time_str):
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H.%M.%S")
    local = ZoneInfo("America/Denver")
    ts_local = ts.replace(tzinfo=local)

    if ts_local.dst() == timedelta(0):
        ts_local += timedelta(hours=1)

    return ts_local