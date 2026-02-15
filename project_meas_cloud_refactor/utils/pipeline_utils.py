import numpy as np
import xarray as xr

def identify_tmin_tmax(xlim, load_xlim, PRF):
    tmin = xlim[0] if load_xlim else 0  # [s]
    tmax = xlim[1] if load_xlim else np.inf  # [s]

    # Set tmin to first shot time if minimum value is 0
    tmin = 1 / PRF if tmin == 0 else tmin

    return tmin, tmax

def check_overlap(ds, tmin, tmax, chunk, file_path, ranges_tot, shots_time_tot, loaded, covered_start, covered_end):
    # Get the first and last time values (assume shots_time is 1D and sorted)
    t0 = ds['shots_time'].isel(shots_time=0).item()
    t1 = ds['shots_time'].isel(shots_time=-1).item()

    overlaps = (t1 >= tmin) and (t0 <= tmax)
    # Check for overlap
    if overlaps:
        print(f'\nIncluding chunk #{chunk}: {t0:.2f}–{t1:.2f}s overlaps {tmin:.2f}–{tmax:.2f}s')
        # Load the full dataset only now
        ds_full = xr.open_dataset(file_path)
        ranges_tot.append(ds_full['ranges'])
        shots_time_tot.append(ds_full['shots_time'])
        print('File loaded.')
        loaded += 1

        # Update coverage flags
        if t0 <= tmin:
            covered_start = True
        if t1 >= tmax:
            covered_end = True

        # Stop early if full range is covered
        if covered_start and covered_end:
            print(f'\nFull time range {tmin:.2f}–{tmax:.2f}s covered by loaded chunks.')
            return 0, loaded, covered_start, covered_end
        else:
            return 1, loaded, covered_start, covered_end
    else:
        print(f'Skipping chunk #{chunk}: {t0:.2f}–{t1:.2f}s not in range {tmin:.2f}–{tmax:.2f}s')
        return 2, loaded, covered_start, covered_end