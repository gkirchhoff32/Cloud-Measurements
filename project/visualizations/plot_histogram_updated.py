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

from load_ARSENL_data import load_INPHAMIS_data, set_binwidth, data_dir, fname, picklename

start = time.time()
# Constants
c = 299792458  # [m/s] Speed of light
UNWRAP_MODULO = 33554431  # max clock count rate 2^25-1

# Parameters
create_csv = False  # Set TRUE to generate a .csv from .ARSENL data
load_data = True  # Set TRUE to load data into a DataFrame and serialize into a pickle object
#exclude = [27500, 33500]  # [ps] Set temporal boundaries for binning
exclude = [0, 70e6]  # [ps] temporal binning bounds
binsize = 12e-8  # [s] bin width for plotting

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
n_shots = len(sync)

# Clean up data first
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
ref_sync = np.zeros((1, len(detect)))  # initialize reference sync timestamps
counts = np.diff(sync_idx) - 1
remainder = detect.index[-1] - sync_idx[-1]
counts = np.append(counts, remainder)
sync_ref = np.repeat(sync_times, counts)

detect_times_rel = detect_times.to_numpy() - sync_ref.to_numpy()
rollover_idx = np.where(detect_times_rel < 0)[0]
detect_times_rel[rollover_idx] += UNWRAP_MODULO
print(np.where(detect_times_rel<0)[0])
print(np.max(detect_times_rel)*25/1e12*3e8/2)
# print()

quit()

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



sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event

detect_time = detect['dtime'].to_numpy()
sync_detect_time = sync_detect['dtime'].to_numpy()

flight_time = (detect_time - sync_detect_time) * 25  # [ps] Time is in segments of 25 ps
flight_time = flight_time[np.where((flight_time >= exclude[0]) & (flight_time < exclude[1]))]  # Exclude t.o.f. where bins ~= 0
distance = flight_time / 1e12 * c / 2

### Histogram of time of flight ###
fig = plt.figure()
ax1 = fig.add_subplot(111)
bin_array = set_binwidth(exclude[0]*1e-12, exclude[1]*1e-12, binsize)
n, bins = np.histogram(flight_time*1e-12, bins=bin_array)
print('Histogram time elapsed: {:.3} sec'.format(time.time() - start))
binwidth = np.diff(bins)[0]
N = n / binwidth / n_shots
center = 0.5 * (bins[:-1]+bins[1:])
ax1.bar(center*1e6, N, align='center', width=binwidth*1e6, color='b', alpha=0.75)
ax1.set_xlabel('Time of flight [us]')
ax1.set_ylabel('Arrival rate [Hz]')
ax1.set_xlim([0, 60])
ax1.set_title('Time of flight for INPHAMIS backscatter')
plt.tight_layout()
plt.show()

