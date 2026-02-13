import numpy as np

def flatten_ranges_shots(ranges_tot, shots_time_tot):
    ranges = np.concatenate([da.values.ravel() for da in ranges_tot])
    shots_time = np.concatenate([da.values.ravel() for da in shots_time_tot])

    return ranges, shots_time