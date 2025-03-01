#
# ARSENL Backscatter Experiments
# load_ARSENL_data.py
#
# Grant Kirchhoff
# Rev. 1: 02-25-2022 Initial loading from raw data
# Rev. 2: 06-24-2022 Loading from netcdf format
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
import xarray as xr
import time
import pickle
import yaml

# Path settings
# NOTE: User should check these settings when running "load_ARSENL_data.py" or "plot_histogram.py"
cwd = os.getcwd()
data_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\SPCM\Data\SPCM_Data_2023.02.06'
fname = r'\Dev_0_-_2023-02-06_17.51.44_OD3.8.ARSENL'
picklename = 'pickle.dat'
create_csv = False

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_INPHAMIS_data(data_dir, fname, picklename, create_csv):
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

    if create_csv:
        start = time.time()
        df.to_csv('{}{}'.format(data_dir + fname, '.csv'), index=None)
        print('Elapsed time (create csv): {} sec'.format(time.time()-start))

    outfile = open('{}/{}'.format(data_dir, picklename), 'wb')
    pickle.dump(df, outfile)
    outfile.close()

    return df

def load_xarray_dataset(data_dir, fname):
    dataset = xr.open_dataset(data_dir + fname)

def set_binwidth(tmin, tmax, resolution):
    bin_array = np.arange(tmin, tmax, resolution)
    return bin_array


if __name__ == "__main__":
    # load_INPHAMIS_data(data_dir, fname, picklename, create_csv)
    cwd = os.getcwd()
    data_loc = cwd + r'/../Data/Deadtime_Experiments_HiFi/Dev_0_-_2022-04-15_10.49.58.ARSENL.OD00.ARSENL.nc'

    dataset = xr.open_dataset(data_loc)

    print(dataset)



