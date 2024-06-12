# deadtime_processing.py
#
# Grant Kirchhoff

"""
Process photon count data using deadtime-fitting routine

IMPORTANT: Make sure to edit the parameters in the "PARAMETERS" section before running.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import time
import pandas as pd
import pickle
import xarray as xr

start = time.time()

cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)
from pathlib import Path

import fit_polynomial_utils_cloud as fit
import data_organize as dorg
from load_ARSENL_data import set_binwidth

########################################################################################################################

### CONSTANTS ####
c = 2.99792458e8  # [m/s] Speed of light

# EDIT THESE PARAMETERS BEFORE RUNNING!
### PARAMETERS ###
exclude_shots = True  # Set TRUE to exclude data to work with smaller dataset (enables 'max_lsr_num_fit_ref' variables)
max_lsr_num_fit = 399  # Maximum number of laser shots for the fit dataset
include_deadtime = True  # Set TRUE to include deadtime in noise model
use_sim = True  # Set TRUE if using simulated data
repeat_run = False  # Set TRUE if repeating processing with same parameters but with different data subsets (e.g., fit number is 1e3 and processing first 1e3 dataset, then next 1e3 dataset, etc.)
# repeat_range = np.arange(1, 13)  # If 'repeat_run' is TRUE, these are the indices of the repeat segments (e.g., 'np.arange(1,3)' and 'max_lsr_num_fit=1e2' --> run on 1st-set of 100, then 2nd-set of 100 shots.
discrete_loss = False  # Set TRUE if using the discrete histogram form of the loss function. Set FALSE if using the time-tag form.
use_muller = False  # Set TRUE if using Muller correction for deadtime correction. Set FALSE if using Deadtime Correction Technique.
if use_muller:
    discrete_loss = True

window_bnd = np.array([975, 1050])  # [m] Set boundaries for binning to exclude outliers
window_bnd = window_bnd / c * 2  # [s] Convert from range to tof
deadtime = 29.1e-9  # [s]
dt = 25e-12  # [s] TCSPC resolution
downsamp = 50  # downsample factor for processing (not when plotting histograms)
downsamp_hist = True  # set TRUE if averaging histograms for ONLY the plotting

# Optimization parameters
rel_step_lim = 1e-8  # termination criteria based on step size
max_epochs = 10000  # maximum number of iterations/epochs
learning_rate = 1e-1  # ADAM learning rate
term_persist = 20  # relative step size averaging interval in iterations

# Polynomial orders (min and max) to be iterated over in specified step size in the optimizer
# Example: Min order 7 and Max order 10 would iterate over orders 7, 8, and 9
M_min = 11
M_max = 24
step = 1
M_lst = np.arange(M_min, M_max, step)

if not repeat_run:
    repeat_range = np.array([1])

### PATH VARIABLES ###
home = str(Path.home())
load_dir = home + r'\OneDrive - UCB-O365\ARSENL\Experiments\Cloud Measurements\Sims\saved_sims'  # Where the data is loaded from
save_dir = load_dir + r'\..\evaluation_loss'  # Where the evaluation loss outputs will be saved
# fname_ref = r'\OD50_Dev_0_-_2023-03-06_16.56.00_OD5.0.ARSENL.nc'  # The dataset that will serve as the high-fidelity reference when evaluating

fname_LG = r'\simnum_0_nshot5.00E+02_useHGFalse_T0.05.nc'
fname_HG = r'\simnum_0_nshot5.00E+02_useHGTrue_T0.95.nc'
sim_num = int(fname_LG.split('_')[1])

# if run_full and use_final_idx:
#     if use_sim:
#         start_idx = 1
#         stop_idx = len(files)
#     else:
#         start_idx = 0
#         stop_idx = len(files) - 1

# Save file name for important outputs (to csv and pickle object). These are used by scripts like "plot_eval_loss.ipynb"
save_csv_file = r'\eval_loss_dtime{}_simnum{}_order{}-{}_shots{:.2E}.csv'.format(include_deadtime, sim_num, M_min, M_max-1, max_lsr_num_fit)
save_csv_file_fit = r'\eval_loss_dtime{}_simnum{}_order{}-{}_shots{:.2E}_best_fit.csv'.format(include_deadtime, sim_num, M_min, M_max-1, max_lsr_num_fit)
save_dframe_fname = r'\fit_figures\eval_loss_dtime{}_simnum{}_order{}-{}' \
                     '_shots{:.2E}_best_fit.pkl'.format(include_deadtime, sim_num, M_min, M_max-1, max_lsr_num_fit)
save_dframe_plot_muller = r'\fit_figures\muller_out_simnum{}_downsamp{}_shots{:.2E}.pkl'.format(sim_num, downsamp, max_lsr_num_fit)
save_dframe_plot_DCT = r'\fit_figures\DCT_out_dtime{}_simnum{}_order{}-{}_shots{:.2E}.pkl'.format(include_deadtime, sim_num, M_min, M_max-1, max_lsr_num_fit)


########################################################################################################################

# I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
t_min = window_bnd[0]  # [s]
t_max = window_bnd[1]  # [s]
if discrete_loss:
    t_fine = np.arange(t_min, t_max, dt*downsamp)  # [s]
    print('Time res: {} s'.format(dt*downsamp))
else:
    t_fine = np.arange(t_min, t_max, dt)  # [s]
    print('Time res: {} s'.format(dt))

# Fitting routine
# for j in range(len(repeat_range)):
val_final_loss_lst = []
percent_active_LG_lst = []
percent_active_HG_lst = []
fit_rate_seg_lst = []
flight_time_lst_LG = []
flight_time_lst_HG = []
active_ratio_hst_lst = []

# Obtain the OD value from the file name. Follow the README guide to ascertain the file naming convention
flight_time_LG, n_shots, t_det_lst_LG, T_BS_LG = dorg.data_organize(dt, load_dir, fname_LG, window_bnd, max_lsr_num_fit, exclude_shots)
flight_time_HG, __, t_det_lst_HG, T_BS_HG = dorg.data_organize(dt, load_dir, fname_HG, window_bnd, max_lsr_num_fit, exclude_shots)

flight_time_LG = flight_time_LG.values
flight_time_HG = flight_time_HG.values

num_det_LG = len(flight_time_LG)
num_det_HG = len(flight_time_HG)
print('Number of detections low gain: {}'.format(len(flight_time_LG)))
print('Number of detections high gain: {}'.format(len(flight_time_HG)))
print('Number of laser shots: {}'.format(n_shots))

if use_sim:
    ds_LG = xr.open_dataset(load_dir + fname_LG)

    photon_rate_arr_LG = ds_LG.photon_rate_arr.to_numpy()
    og_t_min, og_t_max = ds_LG.t_sim_bins.values[0], ds_LG.t_sim_bins.values[-1]
    og_t_fine = np.arange(og_t_min, og_t_max, dt)  # [s]
    idx_min = np.argmin(abs(og_t_fine - t_min))

    if use_muller or discrete_loss:
        photon_rate_arr_LG = photon_rate_arr_LG[idx_min:idx_min + len(t_fine) * downsamp]
        photon_rate_arr_LG = photon_rate_arr_LG[::downsamp]
    else:
        photon_rate_arr_LG = photon_rate_arr_LG[idx_min:idx_min + len(t_fine)]

if not use_muller:
    try:
        t_phot_fit_tnsr_LG, t_phot_val_tnsr_LG, \
        t_det_lst_fit_LG, t_det_lst_val_LG, n_shots_fit_LG, \
        n_shots_val_LG = fit.generate_fit_val(flight_time_LG, t_det_lst_LG, n_shots)

        t_phot_fit_tnsr_HG, t_phot_val_tnsr_HG, \
        t_det_lst_fit_HG, t_det_lst_val_HG, n_shots_fit_HG, \
        n_shots_val_HG = fit.generate_fit_val(flight_time_HG, t_det_lst_HG, n_shots)
    except:
        ZeroDivisionError
        print('ERROR: Insufficient laser shots... increase the "max_lsr_num_fit" parameter.')
        exit()

    # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
    intgrl_N = len(t_fine)
    if not include_deadtime:
        bin_edges = np.linspace(t_min, t_max, intgrl_N + 1, endpoint=False)

        active_ratio_hst_fit_LG = torch.ones(len(bin_edges) - 1)
        active_ratio_hst_val_LG = torch.ones(len(bin_edges) - 1)

        active_ratio_hst_fit_HG = torch.ones(len(bin_edges) - 1)
        active_ratio_hst_val_HG = torch.ones(len(bin_edges) - 1)
    else:
        active_ratio_hst_fit_LG = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_fit_LG, n_shots_fit_LG)
        active_ratio_hst_val_LG = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_val_LG, n_shots_val_LG)

        active_ratio_hst_fit_HG = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_fit_HG, n_shots_fit_HG)
        active_ratio_hst_val_HG = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_val_HG, n_shots_val_HG)

    percent_active_LG = torch.sum(active_ratio_hst_fit_LG).item()/len(active_ratio_hst_fit_LG)
    percent_active_LG_lst.append(percent_active_LG)
    percent_active_HG = torch.sum(active_ratio_hst_fit_HG).item()/len(active_ratio_hst_fit_HG)
    percent_active_HG_lst.append(percent_active_HG)

    t_phot_fit_tnsr = [t_phot_fit_tnsr_LG, t_phot_fit_tnsr_HG]
    t_phot_val_tnsr = [t_phot_val_tnsr_LG, t_phot_val_tnsr_HG]
    active_ratio_hst_fit = [active_ratio_hst_fit_LG, active_ratio_hst_fit_HG]
    active_ratio_hst_val = [active_ratio_hst_val_LG, active_ratio_hst_val_HG]
    n_shots_fit = [n_shots_fit_LG, n_shots_fit_HG]
    n_shots_val = [n_shots_val_LG, n_shots_val_HG]
    T_BS = [T_BS_LG, T_BS_HG]

    # Run fit optimizer
    ax, val_loss_arr, \
    fit_rate_fine, coeffs = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr,
                                   active_ratio_hst_fit, active_ratio_hst_val, n_shots_fit,
                                   n_shots_val, T_BS, discrete_loss, learning_rate, rel_step_lim, intgrl_N, max_epochs,
                                   term_persist)

    ax.set_ylabel('Loss')
    ax.set_xlabel('Iterations')
    ax.set_title('Simulation Number {}'.format(sim_num))
    plt.suptitle('Fit loss')
    plt.tight_layout()
    ax.legend()

    print('Validation loss for\n')
    for i in range(len(M_lst)):
        print('Order {}: {:.5f}'.format(M_lst[i], val_loss_arr[M_lst[i]]))

    # Choose order to investigate
    minx, miny = np.nanargmin(val_loss_arr), np.nanmin(val_loss_arr)
    min_order = minx
    try:
        model = coeffs[min_order, 0:min_order+1]
        for i in range(min_order+1):
            print('Final C{}: {:.4f}'.format(i, model[i]))
    except:
        print("\nERROR: Order exceeds maximum complexity iteration value.\n")

    val_final_loss_lst.append(val_loss_arr[min_order])

    # Arrival rate fit
    fit_rate_seg = fit_rate_fine[min_order, :]

print('Total run time: {} seconds'.format(time.time()-start))

####### Export to CSV and PKL files #######

if not use_muller:
    fit_rate_seg_lst.append(fit_rate_seg)
    flight_time_lst_LG.append(flight_time_LG)
    flight_time_lst_HG.append(flight_time_HG)
    active_ratio_hst_lst.append(active_ratio_hst_fit)

    # Save to csv file
    headers = ['Avg %-age Dectector Active LG', 'Avg %-age Dectector Active HG']
    if use_sim:
        df_out = pd.concat([pd.DataFrame(percent_active_LG_lst), pd.DataFrame(percent_active_HG_lst)], axis=1)
    else:
        df_out = pd.concat([pd.DataFrame(percent_active_LG_lst), pd.DataFrame(percent_active_HG_lst)], axis=1)
    save_csv_file_temp = save_csv_file
    save_csv_file_fit_temp = save_csv_file_fit
    save_dframe_fname_temp = save_dframe_fname

    df_out = df_out.to_csv(save_dir + save_csv_file_temp, header=headers)

    headers = ['time vector', 'fit profile']
    df_out = pd.DataFrame(np.array(fit_rate_seg_lst).T.tolist())
    df_out = pd.concat([pd.DataFrame(t_fine), df_out], axis=1)
    df_out = df_out.to_csv(save_dir + save_csv_file_fit_temp, header=headers)

    dframe = [flight_time_lst_LG, flight_time_lst_HG, t_min, t_max, dt, n_shots, active_ratio_hst_fit_LG, active_ratio_hst_fit_HG]
    pickle.dump(dframe, open(save_dir+save_dframe_fname_temp, 'wb'))

    save_dframe_fname_temp = save_dframe_plot_DCT
    dframe = [discrete_loss, dt, downsamp, t_min, t_max, dt, flight_time_LG, flight_time_HG, n_shots, T_BS_LG, photon_rate_arr_LG, sim_num, t_fine, fit_rate_seg, min_order]
    pickle.dump(dframe, open(save_dir + save_dframe_fname_temp, 'wb'))
else:
    save_dframe_fname_temp = save_dframe_plot_muller
    dframe = [dt, downsamp, t_min, t_max, flight_time_LG, flight_time_HG, n_shots, deadtime, T_BS_LG, T_BS_HG, photon_rate_arr_LG, sim_num, t_fine]
    pickle.dump(dframe, open(save_dir + save_dframe_fname_temp, 'wb'))

###### Plot figures ######
if not repeat_run:
    if not use_muller:
        # Plotting error
        if discrete_loss:
            res = dt * downsamp  # [s]
        else:
            res = dt  # [s]
        print('Processed Resolution: {} m ({} s)'.format(res * c / 2, res))
        bin_array = set_binwidth(t_min, t_max+res, res)
        n_LG, bins = np.histogram(flight_time_LG, bins=bin_array)
        binwidth = np.diff(bins)[0]
        N_LG = n_LG / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        photon_rate_arr = photon_rate_arr_LG / T_BS_LG
        LG_error = np.abs(photon_rate_arr - N_LG / T_BS_LG)  # [Hz] Absolute error with scaled LG histogram

        fig = plt.figure(figsize=(8, 4), dpi=400)
        ax = fig.add_subplot(111)
        ax.plot(t_fine * c / 2, LG_error, color='green', linestyle='--', label='Scaled LG error')
        ax.plot(t_fine * c / 2, np.abs(photon_rate_arr - fit_rate_seg), color='r', linestyle='--', label='Fit error')
        ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Absolute Error [Hz]')
        ax.tick_params(axis='y', which='minor')
        ax.semilogy()
        ax.set_ylim([1e4, 1e9])
        plt.legend(prop={'size': 6})
        plt.tight_layout()

        # Plotting histograms
        use_bins = True
        if use_bins:
            dsamp = 500  # number of bins to downsample when plotting
            res_plot = dt * dsamp
        else:
            res_plot = 2  # [m]
            res_plot = int(res_plot/c*2 // dt) * dt  # [s]
        print('Figure Resolution: {} m ({} s)'.format(res_plot * c / 2, res_plot))
        bin_array = set_binwidth(t_min, t_max+res_plot, res_plot)
        n_LG, bins = np.histogram(flight_time_LG, bins=bin_array)
        n_HG, __ = np.histogram(flight_time_HG, bins=bin_array)
        binwidth = np.diff(bins)[0]
        N_LG = n_LG / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        N_HG = n_HG / binwidth / n_shots  # [Hz]
        # try accomodating for combined high-gain and low-gain signals
        photon_rate_arr = photon_rate_arr_LG / T_BS_LG
        center = 0.5 * (bins[:-1] + bins[1:])

        fig = plt.figure(figsize=(8, 4), dpi=400)
        ax = fig.add_subplot(111)
        ax.bar(center * c / 2, N_LG / 1e6 / T_BS_LG, align='center', width=binwidth*c/2, color='green', alpha=0.75, label='Low gain counts (scaled)')
        ax.bar(center * c / 2, N_HG / 1e6, align='center', width=binwidth * c / 2, color='blue', alpha=0.75, label='High gain counts')
        ax.bar(center * c / 2, N_LG / 1e6, align='center', width=binwidth * c / 2, color='orange', alpha=0.75, label='Low gain counts')
        ax.plot(t_fine * c / 2, fit_rate_seg / 1e6, color='r', linestyle='--', label='Fit')
        ax.plot(t_fine * c / 2, photon_rate_arr / 1e6, color='m', linestyle='--', label='Truth (BS eta)')
        ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Photon Arrival Rate [MHz]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=6, verticalalignment='top', bbox=props)
        ax.semilogy()
        plt.legend(prop={'size': 6})
        plt.tight_layout()
        plt.show()
    else:
        # Plotting histograms
        res = dt * downsamp  # [s]
        print('Processed Resolution: {:.3f} m ({} s)'.format(res * c / 2, res))
        bin_array = set_binwidth(t_min, t_max, res)
        n_LG, bins = np.histogram(flight_time_LG, bins=bin_array)
        n_HG, __ = np.histogram(flight_time_HG, bins=bin_array)
        binwidth = np.diff(bins)[0]
        N_LG = n_LG / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        N_HG = n_HG / binwidth / n_shots  # [Hz]
        N_LG_muller = N_LG / (1 - N_LG*deadtime)  # [Hz] Muller correction applied directly to (unscaled) histogram
        N_HG_muller = N_HG / (1 - N_HG*deadtime)  # [Hz]
        N_LG_muller = N_LG_muller / T_BS_LG  # [Hz] Rescale histogram
        N_HG_muller = N_HG_muller / T_BS_HG  # [Hz]
        center = 0.5 * (bins[:-1] + bins[1:])  # [s]
        photon_rate_arr = photon_rate_arr_LG / T_BS_LG  # [Hz]

        fig = plt.figure(figsize=(8, 4), dpi=400)
        ax = fig.add_subplot(111)
        ax.bar(center * c / 2, N_LG_muller / 1e6, align='center', width=binwidth * c / 2, color='red', alpha=0.50, label='Muller Correction LG')
        # ax.bar(center * c / 2, N_HG_muller / 1e6, align='center', width=binwidth * c / 2, color='black', alpha=0.50, label='Muller Correction (High Gain)')
        ax.bar(center * c / 2, N_LG / 1e6 / T_BS_LG, align='center', width=binwidth * c / 2, color='green', alpha=0.75, label='Low Gain (Scaled)')
        ax.bar(center * c / 2, N_HG / 1e6, align='center', width=binwidth * c / 2, color='blue', alpha=0.75, label='High Gain')
        ax.bar(center * c / 2, N_LG / 1e6, align='center', width=binwidth * c / 2, color='orange', alpha=0.75, label='Low Gain')
        ax.plot(t_fine * c / 2, photon_rate_arr / 1e6, color='m', linestyle='--', label='Truth')
        ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Photon Arrival Rate [MHz]')
        plt.yscale('symlog')
        plt.legend(prop={'size': 6})
        plt.tight_layout()

        # Plotting error
        LG_error = np.abs(photon_rate_arr[:-1] - N_LG_muller)  # [Hz] Absolute error with Muller corrected histogram
        HG_error = np.abs(photon_rate_arr[:-1] - N_HG_muller)  # [Hz] Absolute error with Muller corrected histogram

        fig = plt.figure(figsize=(8, 4), dpi=400)
        ax = fig.add_subplot(111)
        ax.plot(center * c / 2, LG_error, color='green', linestyle='--', label='Muller LG error')
        ax.plot(center * c / 2, HG_error, color='blue', linestyle='--', label='Muller HG error')
        ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Absolute Error [Hz]')
        ax.tick_params(axis='y', which='minor')
        ax.semilogy()
        ax.set_ylim([1e4, 1e9])
        plt.legend(prop={'size': 6})
        plt.tight_layout()
        plt.show()




