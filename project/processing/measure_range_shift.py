"""
Purpose: Measure range delay between CoBaLT boards on startup.
Dataset: 4000-second measurement of laser flash. Use data in "Timing and Range Tests" directory.
General function: Load datasets from high- and low-gain channels. Integrate flash temporally and detect range shift
between channels down to 0.1 m precision, which is the pulse length.

09.15.2025
Grant Kirchhoff grant.kirchhoff@colorado.edu
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
from pathlib import Path
import matplotlib.gridspec as gridspec

from data_preprocessor import DataPreprocessor

def gen_histogram(dp):
    """
    Description: Generate histogram for these tests. Different from "data_preprocessor.gen_histogram" method because
    there's more control over time and range limits.

    Make sure to set appropriate x and y limits in the config file "preprocessing.yaml". Example values for
    'Dev_0_-_2025-09-11_18.06.18.ARSENL' are "ylim: [12.0e-3, 15.0e-3]" [km] and "xlim: [0, 3800] [s]".

    Params
    dp: DataPreprocessor object. Make sure "preprocess" method has been executed already

    Return
    t_binedges: [s] histogram time bin edges
    r_binedges: [m] histogram range bin edges
    flux_raw: [Hz] histogrammed fluxes (counts / shots / range resolution)
    """

    min_time = dp.xlim[0]  # [s]
    max_time = dp.xlim[1]  # [s]
    min_range = dp.ylim[0] * 1e3  # [m]
    max_range = dp.ylim[1] * 1e3  # [m]

    dp.load_chunk()

    # Ravel "ranges" and "shots_time" lists into one continuous array
    ranges = np.concatenate([da.values.ravel() for da in dp.ranges_tot])
    shots_time = np.concatenate([da.values.ravel() for da in dp.shots_time_tot])
    max_shots_idx = np.argmin(np.abs(shots_time - max_time))
    min_shots_idx = np.argmin(np.abs(shots_time - min_time))
    ranges = ranges[min_shots_idx:max_shots_idx]
    shots_time = shots_time[min_shots_idx:max_shots_idx]

    print('\nStarting to generate histogram...')
    start = time.time()
    tbins = np.arange(min_time, max_time, dp.tbinsize)  # [s]
    rbins = np.arange(min_range, max_range, dp.rbinsize)  # [m]

    # Generate 2D histogram
    H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # [cnts] [s] [m]
    H = H.T  # flip axes
    flux = H / (dp.rbinsize / dp.c * 2) / (
            dp.tbinsize * dp.PRF)  # [Hz] Backscatter flux $\Phi = n/N/(\Delta t)$,
    # where $\Phi$ is flux, $n$ is photon counts, $N$ is laser shots number, and $\Delta t$ is range resolution.

    print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

    return {
        't_binedges': t_binedges,
        'r_binedges': r_binedges,
        'flux_raw': flux
    }

def cross_correlate(sig_1, sig_2):
    """
    calculate cross correlation of two signals
    Inputs:
    sig_1 (nx1): signal one
    sig_2 (nx1): signal two
    Returns:
    xcorr_sig (nx1): correlation output
    """

    eps = 1e-12  # small value to avoid dividing by zero
    Sig_1 = np.fft.fft(sig_1)
    Sig_2 = np.fft.fft(sig_2)
    phase_corr = Sig_1 * np.conj(Sig_2) / (np.abs(Sig_1 * np.conj(Sig_2)) + eps)  # 1D phase correlator
    xcorr_sig = np.fft.ifft(phase_corr)
    xcorr_sig = np.fft.fftshift(xcorr_sig)
    xcorr_sig = np.abs(xcorr_sig)  # can take absolute value since imaginary components are negligible

    return xcorr_sig

def loc_max(xcorr_sig_tot, shift_ranges):
    """
    Detects range shift based on peak of correlation output
    Input:
    xcorr_sig_tot: (nxm) correlated signal output (can be accumulated over m runs)
    shift_ranges [m]: (nx1) ranges associated with correlation shifts
    """
    # Locate maximum in cross-correlation result
    max_idx = np.argmax(xcorr_sig_tot, axis=0)
    max_rshift = shift_ranges[max_idx]

    return max_rshift

def main():
    """
    Start by loading data and creating histograms. Make sure histograms integrate the hard-target region in time over
    the entire acquisition period and keep high range resolution. Look at "gen_histogram" function description.
    """

    config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load high- and low-gain '.ARSENL' data. Preprocess and create histograms.
    dp_hg = DataPreprocessor(config)
    dp_lg = DataPreprocessor(config)
    dp_hg.fname = r'/Dev_0_-_2025-09-11_18.06.18.ARSENL'
    dp_lg.fname = r'/Dev_1_-_2025-09-11_18.06.18.ARSENL'
    dp_hg.preprocess()
    dp_lg.preprocess()
    histogram_results_hg = gen_histogram(dp_hg)
    histogram_results_lg = gen_histogram(dp_lg)

    # Measured photon flux
    g1_orig = histogram_results_hg['flux_raw']
    g2_orig = histogram_results_lg['flux_raw']

    # Start cross-correlation operation and bootstrap resample to generate empirical spread in results.
    n_rbins, n_frames = g1_orig.shape
    nboot = 2000  # Number of runs in bootstrap
    xcorr_sig_val_tot = np.zeros((n_rbins, nboot))
    xcorr_sig_test_tot = np.zeros((n_rbins, nboot))
    for i in range(nboot):
        rand_idx = np.random.permutation(n_frames)
        half_nframes = n_frames // 2
        idx_val = rand_idx[:half_nframes]
        idx_test = rand_idx[half_nframes:]

        g1_val = g1_orig[:, idx_val].sum(axis=1)
        g2_val = g2_orig[:, idx_val].sum(axis=1)
        g1_test = g1_orig[:, idx_test].sum(axis=1)
        g2_test = g2_orig[:, idx_test].sum(axis=1)

        xcorr_sig_val = cross_correlate(g1_val, g2_val)
        xcorr_sig_test = cross_correlate(g1_test, g2_test)

        xcorr_sig_val_tot[:, i] = xcorr_sig_val
        xcorr_sig_test_tot[:, i] = xcorr_sig_test

    # Generate cross-correlation axis
    r_binedges = histogram_results_lg['r_binedges']  # [m]
    dr = r_binedges[1] - r_binedges[0]  # [m]
    r_centers = r_binedges[:-1] + 0.5 * dr  # [m]
    nr = len(r_centers)
    shift_pix = np.arange(-(nr // 2), nr // 2 + 1, 1)
    shift_ranges = shift_pix * dr  # [m]

    # Locate maximum in cross-correlation result
    max_rshift_val = loc_max(xcorr_sig_val_tot, shift_ranges)
    max_rshift_test = loc_max(xcorr_sig_test_tot, shift_ranges)

    # Calculate shift and uncertainty estimate
    residual = max_rshift_val - max_rshift_test
    stdev_rshift = np.sqrt(np.sum(residual**2) / (len(max_rshift_val) - 1))
    mean_rshift_test = np.mean(max_rshift_test)

    # Plot bootstrapped cross-correlation results.
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    fig = plt.figure(figsize=(8, 4), dpi=400)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(max_rshift_val, range(nboot), '.', markersize=1, label='Val')
    ax1.plot(max_rshift_test, range(nboot), '.', markersize=1, label='Test')
    ax1.axvline(x=mean_rshift_test, linestyle='--', label='Mean')
    ax1.axvline(x=mean_rshift_test+stdev_rshift, linestyle='--', color='red', label='$\pm\sigma$')
    ax1.axvline(x=mean_rshift_test-stdev_rshift, linestyle='--', color='red')
    ax1.set_ylabel('Bootstrap Sample')
    ax1.set_xlabel('Estimated Shift [m]')
    ax1.set_title('Shift Estimates\nRange Shift Avg: {:.3f} m\nRange Shift StDev: {:.3f} m'.format(mean_rshift_test, stdev_rshift))
    ax1.legend()
    ax2 = fig.add_subplot(gs[1])
    for i in range(nboot):
        ax2.plot(shift_ranges, xcorr_sig_test_tot[:, i], '.', markersize=1, alpha=0.2)
    ax2.axvline(x=mean_rshift_test, linestyle='--', label='Mean', alpha=0.5)
    ax2.set_xlabel('Range shift [m]')
    ax2.set_ylabel('Correlation strength')
    ax2.set_title('Cross-Correlation: {:.3f} m precision\nRange Shift Average: {:.3f} m'.format(dr, mean_rshift_test))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


