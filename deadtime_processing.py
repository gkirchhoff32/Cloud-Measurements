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
max_lsr_num_fit = 999  # Maximum number of laser shots for the fit dataset
# use_final_idx = True  # Set TRUE if you want to use up to the OD value preceding the reference OD
# start_idx = 5  # If 'use_final_idx' FALSE, set the min idx value to this value (for troubleshooting purposes)
# stop_idx = 6  # If 'use_final_idx' FALSE, set the max+1 idx value to this value (for troubleshooting purposes)
# run_full = True  # Set TRUE if you want to run the fits against all ODs. Otherwise, it will just load the reference data
include_deadtime = True  # Set TRUE to include deadtime in noise model
use_sim = True  # Set TRUE if using simulated data
repeat_run = False  # Set TRUE if repeating processing with same parameters but with different data subsets (e.g., fit number is 1e3 and processing first 1e3 dataset, then next 1e3 dataset, etc.)
# repeat_range = np.arange(1, 13)  # If 'repeat_run' is TRUE, these are the indices of the repeat segments (e.g., 'np.arange(1,3)' and 'max_lsr_num_fit=1e2' --> run on 1st-set of 100, then 2nd-set of 100 shots.

window_bnd = np.array([975, 1050])  # [m] Set boundaries for binning to exclude outliers
window_bnd = window_bnd / c * 2  # [s] Convert from range to tof
deadtime = 29.1e-9  # [s]
dt = 25e-12  # [s] TCSPC resolution

# Optimization parameters
rel_step_lim = 1e-8  # termination criteria based on step size
max_epochs = 10000  # maximum number of iterations/epochs
learning_rate = 1e-1  # ADAM learning rate
term_persist = 20  # relative step size averaging interval in iterations

# Polynomial orders (min and max) to be iterated over in specified step size in the optimizer
# Example: Min order 7 and Max order 10 would iterate over orders 7, 8, and 9
M_min = 11
M_max = 12
step = 1
M_lst = np.arange(M_min, M_max, step)

if not repeat_run:
    repeat_range = np.array([1])

### PATH VARIABLES ###
home = str(Path.home())
load_dir = home + r'\OneDrive - UCB-O365\ARSENL\Experiments\Cloud Measurements\Sims\saved_sims'  # Where the data is loaded from
save_dir = load_dir + r'\..\evaluation_loss'  # Where the evaluation loss outputs will be saved
# fname_ref = r'\OD50_Dev_0_-_2023-03-06_16.56.00_OD5.0.ARSENL.nc'  # The dataset that will serve as the high-fidelity reference when evaluating

fname_LG = r'\simnum_4_nshot1.00E+03_useHGFalse.nc'
fname_HG = r'\simnum_4_nshot1.00E+03_useHGTrue.nc'
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

########################################################################################################################

# I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
t_min = window_bnd[0]  # [s]
t_max = window_bnd[1]  # [s]
t_fine = np.arange(t_min, t_max, dt)  # [s]

intgrl_N = len(t_fine)

# Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector
# was "active vs dead"
bin_edges = np.linspace(t_min, t_max, intgrl_N+1, endpoint=False)

# Fitting routine
# for j in range(len(repeat_range)):
val_final_loss_lst = []
eval_final_loss_lst = []
C_scale_final = []
percent_active_LG_lst = []
percent_active_HG_lst = []
fit_rate_seg_lst = []
flight_time_lst_LG = []
flight_time_lst_HG = []
active_ratio_hst_lst = []

# Obtain the OD value from the file name. Follow the README guide to ascertain the file naming convention
flight_time_LG, n_shots, t_det_lst_LG = dorg.data_organize(dt, load_dir, fname_LG, window_bnd, max_lsr_num_fit, exclude_shots)
flight_time_HG, __, t_det_lst_HG = dorg.data_organize(dt, load_dir, fname_HG, window_bnd, max_lsr_num_fit, exclude_shots)

flight_time_LG = flight_time_LG.values
flight_time_HG = flight_time_HG.values
# flight_time_com = np.concatenate((flight_time_LG, flight_time_HG), axis=None)
# t_det_lst_com = t_det_lst_LG + t_det_lst_HG

num_det_LG = len(flight_time_LG)
num_det_HG = len(flight_time_HG)
print('Number of detections low gain: {}'.format(len(flight_time_LG)))
print('Number of detections high gain: {}'.format(len(flight_time_HG)))
print('Number of laser shots: {}'.format(n_shots))

if use_sim:
    ds_LG = xr.open_dataset(load_dir + fname_LG)

    photon_rate_arr_LG = ds_LG.photon_rate_arr.to_numpy()
    og_t_fine = np.arange(0, 2000/c*2, dt)
    idx_min = np.argmin(abs(og_t_fine - t_min))
    photon_rate_arr_LG = photon_rate_arr_LG[idx_min:idx_min+len(t_fine)]

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
if not include_deadtime:
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

# Run fit optimizer
ax, val_loss_arr, \
fit_rate_fine, coeffs, \
C_scale_arr = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr,
                               active_ratio_hst_fit, active_ratio_hst_val, n_shots_fit,
                               n_shots_val, learning_rate, rel_step_lim, intgrl_N, max_epochs,
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
C_scale_final.append(C_scale_arr[min_order])

# Arrival rate fit
fit_rate_seg = fit_rate_fine[min_order, :]

if not repeat_run:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    res_ideal = 2  # [m]
    res = (res_ideal/c*2 // dt) * dt  # [s]
    bin_avg = int(res / dt)
    print('Figure Resolution: {} m'.format(res*c/2))
    bin_array = set_binwidth(t_min, t_max, res)
    n_LG, bins = np.histogram(flight_time_LG, bins=bin_array)
    n_HG, __ = np.histogram(flight_time_HG, bins=bin_array)
    binwidth = np.diff(bins)[0]
    N_LG = n_LG / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
    N_HG = n_HG / binwidth / n_shots  # [Hz]
    # try accomodating for combined high-gain and low-gain signals
    photon_rate_arr = photon_rate_arr_LG / 0.05
    center = 0.5 * (bins[:-1] + bins[1:])

    muller_res_ideal = 50  # [m]
    muller_res = (muller_res_ideal/c*2 // dt) * dt  # [s]
    muller_bin_avg = int(muller_res / dt)
    print('Muller Resolution: {} m'.format(muller_res*c/2))
    muller_bin_array = set_binwidth(t_min, t_max, muller_res)
    n_LG_muller, bins_muller = np.histogram(flight_time_LG, bins=muller_bin_array)
    n_HG_muller, __ = np.histogram(flight_time_HG, bins=muller_bin_array)
    binwidth_muller = np.diff(bins_muller)[0]
    N_LG_muller = n_LG_muller / binwidth_muller / n_shots  # [Hz] Scaling counts to arrival rate
    N_HG_muller = n_HG_muller / binwidth_muller / n_shots  # [Hz]
    N_LG_muller = N_LG_muller / (1-deadtime*N_LG_muller)  # [Hz] applying Muller correction
    N_HG_muller = N_HG_muller / (1-deadtime*N_HG_muller)  # [Hz] applying Muller correction
    center_muller = 0.5 * (bins_muller[:-1] + bins_muller[1:])

    # ax.bar(center_muller * c / 2, N_HG_muller / 1e6 / 0.424, align='center', width=binwidth_muller*c/2, color='teal', alpha=0.75, label='High gain counts (scaled + Muller)')
    # ax.bar(center_muller * c / 2, N_LG_muller / 1e6 / 0.022, align='center', width=binwidth_muller*c/2, color='magenta', alpha=0.75, label='Low gain counts (scaled + Muller)')
    ax.bar(center * c / 2, N_LG / 1e6 / 0.05, align='center', width=binwidth*c/2, color='green', alpha=0.75, label='Low gain counts (scaled)')
    # ax.bar(center * c / 2, N_HG / 1e6 / 0.95, align='center', width=binwidth*c/2, color='green', alpha=1.0, label='High gain counts (scaled)')
    ax.bar(center * c / 2, N_HG / 1e6, align='center', width=binwidth * c / 2, color='blue', alpha=0.75, label='High gain counts')
    ax.bar(center * c / 2, N_LG / 1e6, align='center', width=binwidth * c / 2, color='orange', alpha=0.75, label='Low gain counts')
    ax.plot(t_fine * c / 2, fit_rate_seg / 1e6, color='r', linestyle='--', label='Fit')
    ax.plot(t_fine * c / 2, photon_rate_arr / 1e6, color='m', linestyle='--', label='Truth (BS eta)')
    ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
    # ax.set_title('Arrival Rate Fit: {}{:.2E}'.format('True Rho = ' if use_sim else 'OD = ', rho_list[k] if use_sim else +OD_list[k]))
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Photon Arrival Rate [MHz]')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.semilogy()
    plt.legend()
    plt.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_fine[int(bin_avg/2):-int(bin_avg/2):bin_avg] * c / 2, np.abs(photon_rate_arr[int(bin_avg/2):-int(bin_avg/2):bin_avg] - N_LG / 0.05), color='green', linestyle='--', label='Scaled LG error')
    # ax.plot(t_fine[int(muller_bin_avg/2):-int(muller_bin_avg/2):muller_bin_avg] * c / 2, np.abs(photon_rate_arr[int(muller_bin_avg/2):-int(muller_bin_avg/2):muller_bin_avg]-N_LG_muller/0.022), color='magenta', linestyle='--', label='LG Muller error')
    # ax.plot(t_fine[int(muller_bin_avg/2):-int(muller_bin_avg/2):muller_bin_avg] * c / 2, np.abs(photon_rate_arr[int(muller_bin_avg/2):-int(muller_bin_avg/2):muller_bin_avg]-N_HG_muller/0.022), color='teal', linestyle='--', label='HG Muller error')
    ax.plot(t_fine * c / 2, np.abs(photon_rate_arr - fit_rate_seg), color='r', linestyle='--', label='Fit error')
    ax.set_title('Arrival-Rate Fit Sim # {}'.format(sim_num))
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Absolute Error [Hz]')
    ax.tick_params(axis='y', which='minor')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.semilogy()
    plt.legend()
    plt.tight_layout()

fit_rate_seg_lst.append(fit_rate_seg)
flight_time_lst_LG.append(flight_time_LG)
flight_time_lst_HG.append(flight_time_HG)
active_ratio_hst_lst.append(active_ratio_hst_fit)

plt.show()



# Save to csv file
headers = ['Evaluation Loss', 'Optimal Scaling Factor', 'Avg %-age Dectector Active LG', 'Avg %-age Dectector Active HG']
if use_sim:
    df_out = pd.concat([pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                        pd.DataFrame(percent_active_LG_lst), pd.DataFrame(percent_active_HG_lst)], axis=1)
else:
    df_out = pd.concat([pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                        pd.DataFrame(percent_active_LG_lst), pd.DataFrame(percent_active_HG_lst)], axis=1)
# if repeat_run:
#     save_csv_file_temp = save_csv_file[:-4] + '_run#{}.csv'.format(repeat_range[j])
#     save_csv_file_fit_temp = save_csv_file_fit[:-4] + '_run#{}.csv'.format(repeat_range[j])
#     save_dframe_fname_temp = save_dframe_fname[:-4] + '_run#{}.pkl'.format(repeat_range[j])
# else:
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

print('Total run time: {} seconds'.format(time.time()-start))

# if not repeat_run:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     if use_sim:
#         ax.semilogx(rho_list[start_idx:stop_idx], eval_final_loss_lst, 'r.')
#     else:
#         ax.plot(OD_list[start_idx:stop_idx], eval_final_loss_lst, 'r.')
#     ax.set_xlabel('{}'.format('Rho [Hz]' if use_sim else 'OD'))
#     ax.set_ylabel('Evaluation loss')
#     ax.set_title('Evaluation Loss vs OD')
#     time.sleep(2)
#     plt.show()



