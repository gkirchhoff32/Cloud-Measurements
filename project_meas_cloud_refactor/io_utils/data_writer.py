import os
import xarray as xr

def write_netCDF(ranges, shots_time, shot_diff, fname_nc, chunk_iter, preprocess_path):
    preprocessed_data = xr.Dataset(
        data_vars=dict(
            ranges=ranges,
            shots_time=shots_time,
            shot_diff=shot_diff
        )
    )
    name, ext = os.path.splitext(fname_nc)
    fname_nc_iter = f"{name}_{chunk_iter}{ext}"
    preprocessed_data.to_netcdf(os.path.join(preprocess_path, fname_nc_iter))