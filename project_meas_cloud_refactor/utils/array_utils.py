import numpy as np
import matplotlib.pyplot as plt

def flatten_ranges_shots(ranges_tot, shots_time_tot):
    ranges = np.concatenate([da.values.ravel() for da in ranges_tot])
    shots_time = np.concatenate([da.values.ravel() for da in shots_time_tot])

    return ranges, shots_time

def bootstrap(H):
    """
    Split histogram columns into train/validation sets by alternating columns.

    H: 2D array [range bins, time bins]
    flux: optional 2D array with same shape as H

    Returns:
        H_train, H_val, flux_train, flux_val
    """
    H_train = np.full(np.shape(H), np.nan)
    H_val = np.full(np.shape(H), np.nan)
    H_train[:, ::2] = H[:, ::2]
    H_val[:, 1::2] = H[:, 1::2]

    return H_train, H_val
    # quit()


    # train_cols = np.arange(H.shape[1]) % 2 == 0  # Even columns
    # val_cols = ~train_cols  # Odd columns
    #
    # H_train = H[:, train_cols]
    # H_val = H[:, val_cols]
    # flux_train = flux[:, train_cols]
    # flux_val = flux[:, val_cols]
    # residual = H_train.shape[1] - H_val.shape[1]
    # if residual > 0:
    #     H_train = H_train[:, :-residual]  # remove leftover column so train and validation lengths match
    #     flux_train = flux_train[:, :-residual]
    #     t_binedges = t_binedges[:-residual]
    # t_train = t_binedges[::2]
    # t_val = t_binedges[1::2]
    #
    # # plt.pcolormesh(t_train, r_binedges, H_train)
    # # plt.show()
    #
    # return H_train, H_val, flux_train, flux_val, t_train, t_val


def subset_binedges(t_binedges, cols):
    idx = np.where(cols)[0]          # selected column indices
    edges = np.unique(
        np.concatenate([t_binedges[idx], t_binedges[idx + 1]])
    )
    return edges
