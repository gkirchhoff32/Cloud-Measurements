import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
from pathlib import Path
import scipy.ndimage as nd
import matplotlib.gridspec as gridspec

from data_preprocessor import DataPreprocessor

config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

range_test = True  # set TRUE to test for range shift. set FALSE to test for time shift

dp_hg = DataPreprocessor(config)
dp_lg = DataPreprocessor(config)
dp_hg.fname = r'/Dev_0_-_2025-09-11_18.06.18.ARSENL'
dp_lg.fname = r'/Dev_1_-_2025-09-11_18.06.18.ARSENL'
dp_hg.preprocess()
dp_lg.preprocess()
histogram_results_hg = dp_hg.gen_histogram()
histogram_results_lg = dp_lg.gen_histogram()

g1_orig = histogram_results_hg['flux_raw']
g2_orig = histogram_results_lg['flux_raw']

n_rbins, n_frames = g1_orig.shape
nboot = 2000
eps = 1e-12  # small value to avoid dividing by zero
xcorr_sig_tot = np.zeros((n_rbins, nboot))
for i in range(nboot):
    rep_idx = np.random.randint(0, n_frames, size=n_frames)
    g1 = g1_orig[:, rep_idx].sum(axis=1)
    g2 = g2_orig[:, rep_idx].sum(axis=1)
    G1 = np.fft.fft(g1)
    G2 = np.fft.fft(g2)
    phase_corr = G1 * np.conj(G2) / (np.abs(G1 * np.conj(G2)) + eps)  # 1D phase correlator
    xcorr_sig = np.fft.ifft(phase_corr)
    xcorr_sig = np.fft.fftshift(xcorr_sig)
    xcorr_sig = np.abs(xcorr_sig)  # can take absolute value since imaginary components are negligible
    xcorr_sig_tot[:, i] = xcorr_sig

if range_test:
    r_binedges = histogram_results_lg['r_binedges']  # [m]
    dr = r_binedges[1] - r_binedges[0]  # [m]
    r_centers = r_binedges[:-1] + 0.5 * dr  # [m]
    nr = len(r_centers)
    shift_pix = np.arange(-(nr // 2), nr // 2 + 1, 1)
    shift_ranges = shift_pix * dr  # [m]

else:
    # Create time axis values
    t_binedges = histogram_results_lg['t_binedges']  # [s] bin edges for time axis of low-gain histogram
    # t_binedges_hg = histogram_results_hg['t_binedges']  # [s] bin edges for time axis of high-gain histogram
    dt = t_binedges[1] - t_binedges[0]  # [s] bin size
    t_centers = t_binedges[:-1] + 0.5 * dt  # [s] centers
    nt = len(t_centers)
    shift_pix = np.arange(-(nt // 2), nt // 2 + 1, 1)
    shift_times = shift_pix * dt  # [s]

# g1 = histogram_results_hg['flux_raw'].squeeze()  # 1D high-gain flux
# g2 = histogram_results_lg['flux_raw'].squeeze()  # 1D low-gain flux
# G1 = np.fft.fft(g1)
# G2 = np.fft.fft(g2)
# eps = 1e-12  # small value to avoid dividing by zero
# phase_corr = G1 * np.conj(G2) / (np.abs(G1 * np.conj(G2)) + eps)  # 1D phase correlator
# xcorr_sig = np.fft.ifft(phase_corr)
# xcorr_sig = np.fft.fftshift(xcorr_sig)
# xcorr_sig = np.abs(xcorr_sig)  # can take absolute value since imaginary components are negligible

# # r_abs is your correlation magnitude map
# window_size = 10  # e.g., 5x5 pixels
# xcorr_sig_smooth = nd.uniform_filter(xcorr_sig, size=window_size)

max_idx = np.argmax(xcorr_sig_tot, axis=0)
max_rshift = shift_ranges[max_idx]
mean_rshift = np.mean(max_rshift)
stdev_rshift = np.std(max_rshift, ddof=1)

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
fig = plt.figure(figsize=(8, 4), dpi=400)
ax1 = fig.add_subplot(gs[0])
ax1.plot(max_rshift, range(nboot), '.', markersize=1)
ax1.axvline(x=mean_rshift, linestyle='--', label='Mean')
ax1.axvline(x=mean_rshift+stdev_rshift, linestyle='--', color='red', label='$\pm\sigma$')
ax1.axvline(x=mean_rshift-stdev_rshift, linestyle='--', color='red')
ax1.set_ylabel('Bootstrap Sample')
ax1.set_xlabel('Estimated Shift [m]')
ax1.set_title('Shift Estimates\nRange Shift Avg: {:.3f} m\nRange Shift StDev: {:.3f} m'.format(mean_rshift, stdev_rshift))
ax1.legend()
ax2 = fig.add_subplot(gs[1])
for i in range(nboot):
    ax2.plot(shift_ranges, xcorr_sig_tot[:, i], '.', markersize=1, alpha=0.2)
ax2.axvline(x=mean_rshift, linestyle='--', label='Mean', alpha=0.5)
ax2.set_xlabel('Range shift [m]')
ax2.set_ylabel('Correlation strength')
ax2.set_title('Cross-Correlation: {:.3f} m precision\nRange Shift Average: {:.3f} m'.format(dr, mean_rshift))
# ax2.legend()
plt.tight_layout()
plt.show()
quit()


# if range_test:
#     max_rshift = shift_ranges[max_idx]
#     print('Peak correlation signal: {} m'.format(max_rshift))
#
#     max_idx_smooth = np.argmax(xcorr_sig_smooth)
#     max_rshift_smooth = shift_ranges[max_idx_smooth]
#     print('Peak correlation signal: {} s'.format(max_rshift_smooth))
# else:
#     max_tshift = shift_times[max_idx]
#     print('Peak correlation signal: {} s'.format(max_tshift))
#
#     max_idx_smooth = np.argmax(xcorr_sig_smooth)
#     max_tshift_smooth = shift_times[max_idx_smooth]
#     print('Peak correlation signal: {} s'.format(max_tshift_smooth))


fig = plt.figure(dpi=400, figsize=(8, 4))
ax = fig.add_subplot(111)
if range_test:
    ax.plot(shift_ranges, xcorr_sig, '-', markersize='2', label='Raw', alpha=0.2)
    ax.plot(shift_ranges, xcorr_sig_smooth, '-', markersize='2', label='Filtered (uniform window {})'.format(window_size))
    ax.set_xlabel('Range shift [m]')
else:
    ax.plot(shift_times, xcorr_sig, '.', markersize='2', label='Raw', alpha=0.2)
    ax.plot(shift_times, xcorr_sig_smooth, '.', markersize='2', label='Filtered (uniform window {})'.format(window_size))
    ax.set_xlabel('Time shift [s]')
ax.set_ylabel('Correlation Strength')
if range_test:
    ax.set_title('High- and Low-Gain Cross-Correlation: {:.3f} m precision\nMax Range Shift (Filtered): {:.3f} m\nMax Range Shift (Unfiltered): {:.3f} m'.format(dr, max_rshift_smooth, max_rshift))
else:
    ax.set_title('High- and Low-Gain Cross-Correlation: {} s precision\nMax Time Shift (Filtered): {} s\nMax Time Shift (Unfiltered: {} s'.format(dt, max_tshift_smooth, max_tshift))

# ax.set_yscale('log')
# ax.set_xlim([-5, 0])
plt.legend()
plt.tight_layout()
plt.show()

