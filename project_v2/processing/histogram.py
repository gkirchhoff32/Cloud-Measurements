import numpy as np
import time

def generate_histogram(
    ranges,
    shots_time,
    rbinsize,
    tbinsize,
    PRF,
    c,
    load_xlim,
    load_ylim,
    xlim,
    ylim,
    active_fraction,
    deadtime,
    gen_hist_bg
):
    """
    Pure function version of histogram generation.

    Returns:
        {
            't_binedges': ...,
            'r_binedges': ...,
            'flux_raw': ...,
            'cnts_raw': ...,
            'reduce_min': ...
        }
    }
    """
    dr_af = rbinsize  # [m]
    dt_af = 1 / PRF  # [s]

    # Round time histogram bin size that factorizes the fine-res bin size
    t_factor = max(1, round(tbinsize / dt_af))
    tbinsize_close = t_factor * dt_af  # [s]
    rbinsize = dr_af  # [m]

    # Set time and range windows
    deadtime_range = deadtime * c / 2  # [m]
    if load_xlim:
        min_time, max_time = xlim[0], xlim[1]  # [s]
    else:
        min_time, max_time = shots_time[0], shots_time[-1]  # [s]
    if load_ylim:
        reduce_min = deadtime_range if active_fraction and (deadtime_range >= rbinsize) else 0
        min_range, max_range = (ylim[0] * 1e3 - reduce_min), (ylim[1] * 1e3)  # [m]
    else:
        reduce_min = 0
        min_range, max_range = 0, (c / 2 / PRF)  # [m]

    if load_xlim:
        max_shots_idx = np.argmin(np.abs(shots_time - max_time))
        min_shots_idx = np.argmin(np.abs(shots_time - min_time))
        shots_time = shots_time[min_shots_idx:max_shots_idx]
        ranges = ranges[min_shots_idx:max_shots_idx]

    if gen_hist_bg:
        print('Using approximate resolutions for background estimate: {:.3e} m x {:.3e} s.'.format(rbinsize, tbinsize_close))
    else:
        print('Using resolutions: {:.3e} m x {:.3e} s.'.format(rbinsize, tbinsize_close))

    rbinsize = rbinsize
    tbinsize = tbinsize_close
    print('Actual range and time bin sizes: {:.3e} m x {:.3e} s'.format(rbinsize, tbinsize))

    if gen_hist_bg:
        print('\nStarting to generate histogram for background estimate...')
    else:
        print('\nStarting to generate histogram...')

    start = time.time()
    tbins = np.arange(shots_time[0], shots_time[-1], tbinsize)  # [s]
    if load_ylim:
        rbins = np.arange(min_range, max_range + rbinsize, rbinsize)  # [m]
    else:
        rbins = np.arange(0, c / 2 / PRF + rbinsize, rbinsize)  # [m]

    # Generate histogram
    H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
    H = H.T  # flip axes
    flux = H / (rbinsize / c * 2) / (tbinsize * PRF)  # [Hz] Backscatter flux

    print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

    return {
        't_binedges': t_binedges,
        'r_binedges': r_binedges,
        'flux_raw': flux,
        'cnts_raw': H,
        'reduce_min': reduce_min
    }
