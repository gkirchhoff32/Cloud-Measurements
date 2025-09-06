


import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import yaml
from pathlib import Path
import xarray as xr
import os
import glob
import scipy.ndimage as nd

from data_preprocessor import DataPreprocessor

config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

dp_hg = DataPreprocessor(config)
dp_lg = DataPreprocessor(config)
dp_hg.fname = r'/Dev_0_-_2025-08-05_02.14.22.ARSENL'
dp_lg.fname = r'/Dev_1_-_2025-08-05_02.14.22.ARSENL'
dp_lg.chunk_start = 0
dp_hg.preprocess()
dp_lg.preprocess()
histogram_results_hg = dp_hg.gen_histogram()
histogram_results_lg = dp_lg.gen_histogram()
# dp_hg.plot_histogram(histogram_results_hg)
# dp_lg.plot_histogram(histogram_results_lg)

flg = histogram_results_lg['flux_raw']  # [Hz] raw low-gain flux value histogrammed
fhg = histogram_results_hg['flux_raw']  # [Hz] raw high-gain flux value histogrammed

# TODO: Zero pad to avoid dimension mismatching

t_binedges = histogram_results_lg['t_binedges']  # [s] bin edges for time axis of histogram
r_binedges = histogram_results_lg['r_binedges']  # [m] bin edges for range axis of histogram
dt = np.diff(t_binedges)[0]  # [s] time resolution
dr = np.diff(r_binedges)[0]  # [m] range resolution
t_centers = t_binedges[:-1] + 0.5 * dt  # [s] values of bin centers for time axis
r_centers = r_binedges[:-1] + 0.5 * dr  # [m] values of bin centers for range axis
nt = len(t_centers)
nr = len(r_centers)

Flg = np.fft.fft2(flg)  # [Hz] DFT of low-gain flux
Fhg = np.fft.fft2(fhg)  # [Hz] DFT of high-gain flux

R = (Flg * np.conj(Fhg)) / (np.abs(Flg * np.conj(Fhg)))  # Phase conjugate

r = np.fft.ifft2(R)
r = np.fft.fftshift(r)

# Create histogram bin axes
t_shift_axis = np.arange(-(nt // 2 + 1), nt // 2 + 1, 1) + 0.5  # [pix] shifted by 0.5 to be the centers of histogram bins
r_shift_axis = np.arange(-(nr // 2 + 1), nr // 2 + 1, 1) + 0.5  # [pix]
t_shift_axis_scale = t_shift_axis * dt  # [s]
r_shift_axis_scale = r_shift_axis * dr  # [m]

max_idx = np.unravel_index(np.argmax(r), r.shape)  # [pix]
print("Peak location (row, col):", max_idx)

# Map to shift in samples (accounting for fftshift)
r_shift_pix = max_idx[0] - nr // 2  # [pix]
t_shift_pix = max_idx[1] - nt // 2  # [pix]

# Convert to physical units
t_shift = t_shift_pix * dt  # [s]
r_shift = r_shift_pix * dr  # [m]

# r_abs = np.abs(r)
r_abs = np.flip(np.abs(r), axis=(0, 1))  # Absolute value and mirror each axis

print('\nPrecisions: Range {:.2e} m and Time {:.5e} s'.format(dr, dt))
print("\nEstimated shift in range: {} pix ({:.2e} m)".format(r_shift_pix, r_shift))
print("\nEstimated shift in time: {} pix  ({:.5e} s)".format(t_shift_pix, t_shift))

# Now smooth the correlation map to denoise

# r_abs is your correlation magnitude map
window_size = 3  # e.g., 5x5 pixels
r_abs_smooth = nd.uniform_filter(r_abs, size=window_size)

max_idx_smooth = np.unravel_index(np.argmax(r_abs_smooth), r_abs_smooth.shape)  # [pix]
print("\nPeak location (row, col):", max_idx_smooth)

# Map to shift in samples (accounting for fftshift)
r_shift_pix_smooth = (max_idx_smooth[0] - nr // 2) * -1  # [pix]
t_shift_pix_smooth = (max_idx_smooth[1] - nt // 2) * -1  # [pix]

# Convert to physical units
t_shift_smooth = t_shift_pix_smooth * dt  # [s]
r_shift_smooth = r_shift_pix_smooth * dr  # [m]

print("\nEstimated shift in range (smoothed): {} pix ({:.2e} m)".format(r_shift_pix_smooth, r_shift_smooth))
print("\nEstimated shift in time (smoothed): {} pix  ({:.5e} s)".format(t_shift_pix_smooth, t_shift_smooth))


fig = plt.figure(dpi=300, figsize=(4, 2))
ax = fig.add_subplot(111)
mesh = ax.pcolormesh(t_shift_axis_scale, r_shift_axis_scale, r_abs, norm=LogNorm(vmin=r_abs.min(), vmax=r_abs.max()))
ax.set_xlabel('Time shift [s]')
ax.set_ylabel('Range shift [m]')
ax.set_title('Phase Correlation (No Smoothing)')
# ax.set_xlim([-2.6, -2.5])
# ax.set_ylim([-1000, 1000])
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Correlation strength')
plt.tight_layout()
plt.show()

fig = plt.figure(dpi=300, figsize=(4, 2))
ax = fig.add_subplot(111)
mesh = ax.pcolormesh(t_shift_axis_scale, r_shift_axis_scale, r_abs_smooth, norm=LogNorm(vmin=r_abs_smooth.min(), vmax=r_abs_smooth.max()))
ax.set_xlabel('Time shift [s]')
ax.set_ylabel('Range shift [m]')
ax.set_title('Phase Correlation (Smoothing)')
# ax.set_xlim([-2.6, -2.5])
# ax.set_ylim([-1000, 1000])
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Correlation strength (smoothed)')
plt.tight_layout()
plt.show()

# fig = plt.figure(dpi=400, figsize=(4, 2))
# ax = fig.add_subplot(111)
# ax.plot(t_shift_axis_scale[1:] - 0.5, np.sum(r_abs, axis=0))
# ax.set_xlabel('Time shift [s]')
# ax.set_ylabel('Correlation strength (time)')
# plt.tight_layout()
# plt.show()
#
# fig = plt.figure(dpi=400, figsize=(4, 2))
# ax = fig.add_subplot(111)
# ax.plot(r_shift_axis_scale[1:] - 0.5, np.sum(r_abs, axis=1))
# ax.set_xlabel('Range shift [m]')
# ax.set_ylabel('Correlation strength (range)')
# plt.tight_layout()
# plt.show()

print('Finished.')
