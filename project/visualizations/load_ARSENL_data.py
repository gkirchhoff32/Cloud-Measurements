#
# ARSENL Backscatter Experiments
# load_ARSENL_data.py
#
# Grant Kirchhoff
# Rev. 1: 02-25-2022 Initial loading from raw data
# Rev. 2: 06-24-2022 Loading from netcdf format
# Updated 07-01-2025 Checking first light from CoBaLT Lidar data
# University of Colorado Boulder
#
"""

Load data from ARSENL INPHAMIS lidar as a DataFrame and serialize as a pickle file. The purpose is to load the large
dataset as a pickle object without having to reload the data every time main() is executed. This reduces execution time
and enables flexibility with the data handling.

"""

import os
import numpy as np
import pandas as pd
import time
import pickle

# Constants
c = 299792458  # [m/s] Speed of light

# data_dir = r'/home/arsenl/inphamis-acquisition/two_picos_v2/CoBaLT_data'
data_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Cloud Measurements\First Light'
fname = r'/Dev_0_-_2025-07-10_01.48.17_180sec.ARSENL'
picklename = 'test'

def load_INPHAMIS_data(data_dir, fname, picklename):
    """
    Loads data from INPHAMIS lidar acquisition system into DataFrame object and stores it as a serialized pickle
    object for fast data loading.
    :param fname: filename to be loaded [str]
    :param picklename: filename of pickle [str]
    :param create_csv: whether or not to create a csv file from the data [bool]
    """

    start = time.time()
    df = pd.read_csv(data_dir + fname, delimiter=',')
    print('Elapsed time (read pd): {} sec'.format(time.time()-start))

    outfile = open('{}/{}'.format(data_dir, picklename), 'wb')
    pickle.dump(df, outfile)
    outfile.close()

    return df

def set_binwidth(tmin, tmax, resolution):
    bin_array = np.arange(tmin, tmax+resolution, resolution)
    return bin_array

def get_times(df, exclude):
    df1 = df.loc[df['dtime'] != 0]
    detect = df1.loc[(df1['overflow'] == 0) & (df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
    sync = df1.loc[(df1['overflow'] == 1) & (df1['channel'] == 0)]
    n_shots = len(sync)
    
    sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
    sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event
    
    detect_time = detect['dtime'].to_numpy()
    sync_detect_time = sync_detect['dtime'].to_numpy()
    
    flight_time = (detect_time - sync_detect_time) * 25  # [ps] Time is in segments of 25 ps
    flight_time = flight_time[np.where((flight_time >= exclude[0]) & (flight_time < exclude[1]))]  # Exclude t.o.f. where bins ~= 0
    distance = flight_time / 1e12 * c / 2

    return flight_time, n_shots



