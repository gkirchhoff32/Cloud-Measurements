#
# ARSENL Backscatter Experiments
# plot_histogram.py
#
# Grant Kirchhoff
# 02-25-2022
# University of Colorado Boulder
#
# Updated: 07-01-2025

"""
Histogram photon arrival time data from ARSENL INPHAMIS lidar. IMPORTANT: Set data path settings in
'load_ARSENL_data.py' first.
"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from load_ARSENL_data import load_INPHAMIS_data, data_dir, fname, picklename

start = time.time()
# Constants
c = 299792458  # [m/s] Speed of light
UNWRAP_MODULO = 33554431  # max clock count rate 2^25-1
PRF = 14.3e3  # [Hz] laser frequency

# Parameters
create_csv = False  # Set TRUE to generate a .csv from .ARSENL data
load_data = False  # Set TRUE to load data into a DataFrame and serialize into a pickle object
histogram = True  # Set TRUE to generate histogram plot, FALSE to generate scatter plot
#exclude = [27500, 33500]  # [ps] Set temporal boundaries for binning
exclude = [0, 70e-6]  # [s] temporal binning bounds
# binsize = 120e-9  # [s] bin width for plotting

# Load INPHAMIS .ARSENL data if not yet serialized
if load_data:
    load_INPHAMIS_data(data_dir, fname, picklename)

# Unpickle the data to DataFrame object
infile = open('{}/{}'.format(data_dir, picklename), 'rb')
df = pickle.load(infile)
infile.close()

df1 = df.loc[df['dtime'] != 0]
detect = df1.loc[(df1['overflow'] == 0) & (df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
sync = df1.loc[(df1['overflow'] == 1) & (df1['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)
rollover = df1.loc[(df1['overflow'] == 1) & (df1['channel'] == 63)]  # Clock rollover ("overflow", "channel" = 1,63) Max count is 2^25-1=33554431

# Clean up data first
print('Starting data conditioning...')
start_clean = time.time()
start_idx = sync.index[0]  # first index where laser is recorded
detect = detect[detect.index > start_idx]  # throw away detections that precede the first laser shot

df2 = df1.drop(rollover.index)  # Remove rollover events
df2 = df2.reset_index(drop=True)  # Reset indices

detect = df2.loc[(df2['overflow'] == 0) & (df2['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
sync = df2.loc[(df2['overflow'] == 1) & (df2['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)
start_idx = sync.index[0]
detect = detect[detect.index > start_idx]

sync_times = sync['dtime']
detect_times = detect['dtime']

sync_idx = sync.index
# ref_sync = np.zeros((1, len(detect)))  # initialize reference sync timestamps
counts = np.diff(sync_idx) - 1
remainder = detect.index[-1] - sync_idx[-1]
counts = np.append(counts, remainder)
sync_ref = np.repeat(sync_times, counts)
shots_ref = np.repeat(np.arange(len(sync)), counts)
shots_time = shots_ref / PRF  # [s] Equivalent time for each shot

detect_times_rel = detect_times.to_numpy() - sync_ref.to_numpy()
rollover_idx = np.where(detect_times_rel < 0)[0]
detect_times_rel[rollover_idx] += UNWRAP_MODULO

flight_time = detect_times_rel * 25e-12  # [s] counts were in 25 ps increments
range = flight_time * c / 2  # [m]

print('Finished data conditioning.\nTime elapsed: {:.1f} seconds'.format(time.time()-start_clean))

if histogram:
    tbinsize = 0.25 # [s]
    rbinsize = 0.1  # [m]

    tbins = np.arange(0, shots_time[-1], tbinsize)  # [s]
    rbins = np.arange(0, c/2/PRF, rbinsize)  # [m]

    H, t_binedges, r_binedges = np.histogram2d(shots_time, range, bins=[tbins, rbins])
    H = H.T  # flip axes
    flux = H / (rbinsize/c*2) / (tbinsize*PRF)  # [Hz]

    bg_edges = [9900, 10100]  # [m] background estimation ranges
    bg_edges_idx = [np.argmin(np.abs(rbins-bg_edges[0])), np.argmin(np.abs(rbins-bg_edges[1]))]
    bg_flux = np.mean(flux[bg_edges_idx[0]:bg_edges_idx[1], :])
    flux_bg_sub = flux - bg_flux  # [Hz] flux with background subtracted

    fig = plt.figure(dpi=200, figsize=(8, 4))
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(t_binedges, r_binedges/1e3, flux_bg_sub, cmap='viridis', norm=LogNorm(vmin=bg_flux, vmax=flux.max()))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Range [km]')
    fig.suptitle('CoBaLT Backscatter Flux')
    ax.set_title('Scale {:.1f} m x {:.2f} s'.format(rbinsize, tbinsize))
    ax.set_ylim([4, 6])  # [km]
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Flux [Hz]')
    plt.tight_layout()
    plt.show()


    # ### Histogram of time of flight ###
    # fig = plt.figure(dpi=300)
    # ax1 = fig.add_subplot(111)
    # bin_array = set_binwidth(exclude[0], exclude[1], binsize)
    # n, bins = np.histogram(flight_time, bins=bin_array)
    # print('Histogram time elapsed: {:.3} sec'.format(time.time() - start))
    # binwidth = np.diff(bins)[0]  # [s]
    # binwidth_range = binwidth * c / 2  # [m]
    # flux = n / binwidth / n_shots  # [Hz]
    # center = 0.5 * (bins[:-1]+bins[1:])  # [s]
    # center_range = center * c / 2  # [m]
    # ax1.barh(center_range/1e3, flux, align='center', height=binwidth_range/1e3, color='b', alpha=0.75)
    # ax1.set_ylabel('Range [km]')
    # ax1.set_xlabel('Arrival rate [Hz]')
    # # ax1.set_xlim([0, 60])
    # ax1.set_title('CoBaLT backscatter')
    # ax1.set_xscale('log')
    # plt.tight_layout()
    # plt.show()


else: # if histogram variable is FALSE, scatter plot
    fig = plt.figure(dpi=200, figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.scatter(shots_time, range/1e3, s=0.001, linewidths=0)
    ax.set_ylim([0, c/2/PRF/1e3])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Range [km]')
    ax.set_title('CoBaLT Backscatter')
    ax.set_ylim([4.5, 5.5])
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.show()



# for i in rollover.index:
#     cnt += 1
#     print(i)
#     # return closest sync that comes after it
#     sync_rollover_idx = sync[sync.index > i].index[0]
#     detect_rollover_idx = detect[(detect.index > i) & (detect.index < sync_rollover_idx)].index
#     rollover_dtime = detect.loc[detect_rollover_idx]['dtime']
#     detect.loc[detect_rollover_idx]['dtime'] = rollover_dtime + max_clock_cnt
#     print(detect_rollover_idx)
#     if cnt == 2:
#         quit()
# #print(rollover)
#
# quit()



# sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
# sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event
#
# detect_time = detect['dtime'].to_numpy()
# sync_detect_time = sync_detect['dtime'].to_numpy()
#
# flight_time = (detect_time - sync_detect_time) * 25  # [ps] Time is in segments of 25 ps
# flight_time = flight_time[np.where((flight_time >= exclude[0]) & (flight_time < exclude[1]))]  # Exclude t.o.f. where bins ~= 0
# distance = flight_time / 1e12 * c / 2

# ### Histogram of time of flight ###
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# bin_array = set_binwidth(exclude[0]*1e-12, exclude[1]*1e-12, binsize)
# n, bins = np.histogram(flight_time*1e-12, bins=bin_array)
# print('Histogram time elapsed: {:.3} sec'.format(time.time() - start))
# binwidth = np.diff(bins)[0]
# N = n / binwidth / n_shots
# center = 0.5 * (bins[:-1]+bins[1:])
# ax1.bar(center*1e6, N, align='center', width=binwidth*1e6, color='b', alpha=0.75)
# ax1.set_xlabel('Time of flight [us]')
# ax1.set_ylabel('Arrival rate [Hz]')
# ax1.set_xlim([0, 60])
# ax1.set_title('Time of flight for INPHAMIS backscatter')
# plt.tight_layout()
# plt.show()

