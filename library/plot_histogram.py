#
# ARSENL Backscatter Experiments
# plot_histogram.py
#
# Grant Kirchhoff
# 02-25-2022
# University of Colorado Boulder
#
"""
Histogram photon arrival time data from ARSENL INPHAMIS lidar. IMPORTANT: Set data path settings in
'load_ARSENL_data.py' first.
"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

from load_ARSENL_data import load_INPHAMIS_data, set_binwidth, data_dir, fname, picklename

start = time.time()
# Constants
c = 299792458  # [m/s] Speed of light

# Parameters
create_csv = False  # Set TRUE to generate a .csv from .ARSENL data
load_data = True  # Set TRUE to load data into a DataFrame and serialize into a pickle object
load_netcdf = True  # Set TRUE if loading from netcdf file ('*.ARSENL.nc'). Set FALSE if loading from *.ARSENL file.
use_donovan = False  # Set TRUE if user wants to scale the histogram by using the Donovan correction
limit_shots = False

# window_bnd = [32e-9, 38e-9]  # [s] Set temporal boundaries for binning
window_bnd = np.array([850, 1200])  # [m] Set boundaries for binning to exclude outliers
window_bnd = window_bnd / c * 2  # [s] Convert from range to tof
dt = 25e-12  # [s] Resolution
deadtime = 29.1e-9  # [s] Deadtime interval (25ns for sim, 29.1ns for SPCM)

if limit_shots:
    use_shots = 250

t_min = window_bnd[0]  # [s]
t_max = window_bnd[1]  # [s]

if load_netcdf:
    home = str(Path.home())
    data_dir = home + r'\OneDrive - UCB-O365\ARSENL\Experiments\Cloud Measurements\Sims\saved_sims'
    fname = r'\\simnum_0_nshot5.00E+02_useHGTrue_T0.95.nc'

    ds = xr.open_dataset(data_dir + fname)

    cnts = ds.time_tag
    if limit_shots:
        lim_shot_idx = (abs(ds.time_tag_sync_index - use_shots)).argmin().values.item()
        cnts = cnts[:lim_shot_idx]

    flight_time = cnts * dt  # [s]
    # Exclude specified t.o.f. bins
    flight_time = flight_time[np.where((flight_time >= window_bnd[0]) & (flight_time < window_bnd[1]))]
    if limit_shots:
        n_shots = len(ds.sync_index[:use_shots])
    else:
        n_shots = len(ds.sync_index)

else:
    # Load INPHAMIS .ARSENL data if not yet serialized
    if load_data:
        load_INPHAMIS_data(data_dir, fname, picklename, create_csv)

    # Unpickle the data to DataFrame object
    infile = open('{}/{}'.format(data_dir, picklename), 'rb')
    df = pickle.load(infile)
    infile.close()

    df1 = df.loc[df['dtime'] != 0]
    detect = df1.loc[(df1['overflow'] == 0) & (df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
    sync = df1.loc[(df1['overflow'] == 1) & (df1['channel'] == 0)]
    n_shots = len(sync)

    sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
    sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event

    detect_time = detect['dtime'].to_numpy()
    sync_detect_time = sync_detect['dtime'].to_numpy()

    flight_time = (detect_time - sync_detect_time) * dt  # [s] Time is in segments of 25 ps
    flight_time = flight_time[np.where((flight_time >= t_min) & (flight_time < t_max))]  # window_bnd t.o.f. where bins ~= 0
    distance = flight_time * c / 2

### Histogram of time of flight ###
fig = plt.figure()
ax1 = fig.add_subplot(111)
bin_avg = 500
res = dt * bin_avg
bin_array = set_binwidth(t_min, t_max, res)
# bin_array = set_binwidth(t_min, t_max, dt)
n, bins = np.histogram(flight_time, bins=bin_array)
print('Histogram plot time elapsed: {:.3} sec'.format(time.time() - start))
binwidth = np.diff(bins)[0]
N = n / binwidth / n_shots
print('Number of shots: {}'.format(n_shots))
if use_donovan:
    N_dono = N / (1 - N*deadtime)
center = 0.5 * (bins[:-1]+bins[1:])
ax1.bar(center*c/2, N, align='center', width=binwidth*c/2, color='b', alpha=0.75, label='Detections')
if use_donovan:
    ax1.bar(center*c/2, N_dono, align='center', width=binwidth*c/2, color='r', alpha=0.5, label='Muller "Corrected" Profile')
    ax1.set_title('Inaccurate Muller Correction Demonstration')
    plt.legend()
ax1.set_xlabel('Range [m]')
ax1.set_ylabel('Arrival rate [Hz]')
ax1.set_yscale('log')
plt.tight_layout()
plt.show()

