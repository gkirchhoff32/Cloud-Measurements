# Method for loading and organizing data necessary for 'fit_histogram_polynomial_complexity_iteration.ipynb'.
#
# Grant Kirchhoff
# Last Updated: 09/21/2022


import numpy as np
import xarray as xr

def data_organize(dt, data_dir, fname, window_bnd, max_lsr_num, exclude_shots=True, repeat_num=1):
    '''
    Some bookkeeping. Organizes data into structures and variables required for the fit routine.
    :param dt: (float) Temporal resolution [s]
    :param data_dir: (str) Data directory
    :param fname: (str) Data file name
    :param window_bnd: (2x1 list) Two time bounds to exclude outlying data [s]
    :param set_max_det: (bool) Choose whether to set the maximum limiter as number of laser shots (0) or number of detection events (1)
    :param max_lsr_num: (int) Number of maximum laser shots to include going forward (see "exclude_shots")
    :param exclude_shots: (bool) Set True if you want to exclude shots beyond the "max_lsr_num" parameter
    :return:
    flight_time: (N_{tags}x1 xarray.DataArray) Time tags [s]
    n_shots: (int) Number of laser shots
    t_det_lst: ( ??xn_shots list) Nested list of xarray.DataArrays, where each DataAarray consists of the detections per laser shot [s]
    '''

    # Load and organize xarray dataset
    ds = xr.open_dataset(data_dir + fname)

    # Generate flight time values, sync counts, and sync counter values for each time tag
    cnts = ds.time_tag
    flight_time = cnts * dt  # [s]
    # Exclude specified t.o.f. bins
    flight_time = flight_time[np.where((flight_time >= window_bnd[0]) & (flight_time < window_bnd[1]))]
    lsr_shot_cntr = ds.sync_index.to_numpy()
    ttag_sync_idx = ds.time_tag_sync_index.values

    if exclude_shots:
        # Obtain sync index corresponding to maximum laser shot number specified by user. If the exact index doesn't
        # have a corresponding detection, then the nearest shot with a detection is chosen.
        max_lsr_num_idx = max_lsr_num * repeat_num
        min_lsr_num_idx = max_lsr_num * (repeat_num-1)
        excl_sync_max = ds.sync_index[max_lsr_num_idx].item()
        excl_sync_min = ds.sync_index[min_lsr_num_idx].item()
        excl_ttag_idx_max = np.where(ttag_sync_idx == excl_sync_max)[0]
        excl_ttag_idx_min = np.where(ttag_sync_idx == excl_sync_min)[0]
        if excl_ttag_idx_max.size == 0:
            nearest_max = ttag_sync_idx[np.argmin(np.abs(ttag_sync_idx - excl_sync_max))] - lsr_shot_cntr[0]  # Subtract first index value to start at 0
            print(
                "Last sync event doesn't correspond to a detection event. Choosing nearest corresponding sync event (index: {})...".format(
                    nearest_max))
            excl_sync_max = ds.sync_index[nearest_max].item()
            excl_ttag_idx_max = np.where(ttag_sync_idx == excl_sync_max)[0][0]
            max_lsr_num_idx = nearest_max
        else:
            excl_ttag_idx_max = excl_ttag_idx_max[0]
        if excl_ttag_idx_min.size == 0:
            nearest_min = ttag_sync_idx[np.argmin(np.abs(ttag_sync_idx - excl_sync_min))] - lsr_shot_cntr[0]  # Subtract first index value to start at 0
            excl_sync_min = ds.sync_index[nearest_min].item()
            excl_ttag_idx_min = np.where(ttag_sync_idx == excl_sync_min)[0][0]
            min_lsr_num_idx = nearest_min
        else:
            excl_ttag_idx_min = excl_ttag_idx_min[0]

        lsr_shot_cntr = lsr_shot_cntr[min_lsr_num_idx:max_lsr_num_idx]
        flight_time = flight_time[excl_ttag_idx_min:excl_ttag_idx_max]
        n_shots = len(lsr_shot_cntr)
    else:
        n_shots = len(ds.sync_index)

    # Generate nested list of DataArrays, where each array consists of the detections per laser shot
    t_det_lst = []
    for i in range(len(flight_time)):
        total_det = ds.time_tag[i].values
        total_det = total_det * dt  # Convert from 25ps intervals to seconds
        total_det = np.array(total_det, ndmin=1, copy=False)
        # If there are more than one detection per laser event, then append those to the same row in t_det_lst. Otherwise just append to a new row like normal.
        if ttag_sync_idx[i] == ttag_sync_idx[i-1]:
            t_det_lst[-1] = np.append(np.array([t_det_lst[-1]]), total_det)
        else:
            t_det_lst.append(total_det)

    return flight_time, n_shots, t_det_lst