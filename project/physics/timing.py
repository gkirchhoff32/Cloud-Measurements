from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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


def adjust_daylight_savings(date_str, time_str):
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H.%M.%S")
    local = ZoneInfo("America/Denver")
    ts_local = ts.replace(tzinfo=local)

    if ts_local.dst() == timedelta(0):
        ts_local += timedelta(hours=1)

    return ts_local
