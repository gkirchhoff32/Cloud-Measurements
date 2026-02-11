# generate_sim_data.py
#
# Grant Kirchhoff
"""
Script to generate simulated retrievals and output data products that can be used in the deadtime noise model fitting
routine (e.g., evaluation_high_OD_iterate.py)
"""

import os, sys
import numpy as np
import time
import matplotlib.pyplot as plt
import xarray as xr

# import library for simulating Poisson point processes
cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)

import sim_deadtime_utils as sim
from load_ARSENL_data_utils import set_binwidth

def gen_sim_data(photon_rate_arr, t_sim_bins, tD, Nshot, wrap_deadtime):
    """
    Using Matthew Hayman's 'photon_count_generator' method in 'sim_deadtime_utils', generate simulated data with
    and without deadtime effects.
    :param t_sim_max: (float) maximum time for each laser shot [s]
    :param dt_sim: (float) resolution settings [s]
    :param tD: (float) deadtime [s]
    :param Nshot: (int) number of laser shots
    :param wrap_deadtime: (bool) set TRUE to wrap deadtime into next shot if detection is close to 't_sim_max'
    :param window_bnd: (1x2 float list) time bounds on simulation [s]
    :param laser_pulse_width: laser pulse width (Gaussian) [s]
    :param target_time: target location in time [s]
    :param target_amplitude: target amplitude peak count rate [Hz]
    :param background: background count rate [Hz]
    :return: flight_time, true_flight_time, n_shots, t_det_lst, t_phot_lst
    """
    ##### GENERATE SIMULATED DATA #####

    dt_sim = np.diff(t_sim_bins)[0]

    # # simulation resolution settings
    # t_sim_min = 0
    #
    # # generate the simulated scene time axis
    # t_sim = np.arange(t_sim_min, t_sim_max, dt_sim)           # simulation time
    # t_min = t_sim_bins[0]  # [s] beginning time window value (track for shifting axis)
    t_sim_bins = np.concatenate((t_sim_bins, t_sim_bins[-1:]+dt_sim))  # simulation time histogram bins
    # t_sim_bins_shifted = t_sim_bins - t_sim_bins[0]  # shifts axis to start at zero, necessary for simulator

    # generate photon counts

    # lists of photon arrivals per laser shot
    start = time.time()
    sync_idx = np.arange(Nshot)  # sync value
    det_sync_idx = []
    phot_sync_idx = []
    det_events = []
    phot_events = []

    t_det_last = -100.0  # last photon detection event
    for n in range(Nshot):
        # simulate a laser shot
        ptime, ctime = sim.photon_count_generator(t_sim_bins, photon_rate_arr, tau_d_flt=tD, last_photon_flt=t_det_last)
        if wrap_deadtime:
            if len(ctime) > 0:
                t_det_last = ctime[-1]
            t_det_last -= t_sim_bins[-1]

        # ctime += t_min  # [s] shift back to actual time value
        # ptime += t_min  # [s] shift back to actual time value

        ctime /= dt_sim  # convert from s to clock counts since sync event
        ptime /= dt_sim  # convert from s to clock counts since sync event

        for i in range(len(ctime)):
            det_events.append(ctime[i])  # detection time tags
            det_sync_idx.append(n)
        for i in range(len(ptime)):
            phot_events.append(ptime[i])  # photon time tags
            phot_sync_idx.append(n)

    det_idx = np.arange(len(det_events))
    phot_idx = np.arange(len(phot_events))

    print('time elapsed: {}'.format(time.time() - start))

    return {
        'det_idx': det_idx,
        'phot_idx': phot_idx,
        'sync_idx': sync_idx,
        'det_sync_idx': det_sync_idx,
        'phot_sync_idx': phot_sync_idx,
        'det_events': det_events,
        'phot_events': phot_events,
        't_sim_bins': t_sim_bins
    }


# if __name__ == '__main__':
#
#     import data_organize as dorg
#
#     ### PARAMETERS ###
#
#     # simulation resolution settings
#     t_sim_min = 0  # [s]
#     t_sim_max = 40e-9  # [s]
#     dt_sim = 25e-12  # [s]
#
#     tD = 25e-9  # [s] deadtime
#     Nshot = int(1e6)  # number of laser shots
#     wrap_deadtime = True  # wrap deadtime between shots
#     window_bnd = [26e-9, 34e-9]  # [s] time-of-flight bounds
#     laser_pulse_width = 500e-12  # [s] laser pulse width
#     target_time = 31.2e-9  # [s] target location in time
#     target_amplitude = 1e7  # [Hz] target peak count rate
#     background = 1e4  # [Hz] background count rate (dark count)
#
#     ### GENERATE SIMULATED DATA ###
#     det_idx, phot_idx, sync_idx, det_sync_idx, phot_sync_idx, det_events, phot_events = gen_sim_data(t_sim_max, dt_sim,
#                                                                                                      tD, Nshot,
#                                                                                                      wrap_deadtime,
#                                                                                                      window_bnd,
#                                                                                                      laser_pulse_width,
#                                                                                                      target_time,
#                                                                                                      target_amplitude,
#                                                                                                      background)
#
#     time_tag_index = det_idx
#     true_time_tag_index = phot_idx
#     sync_index = sync_idx
#     time_tag = det_events
#     true_time_tag = phot_events
#     time_tag_sync_index = det_sync_idx
#     true_time_tag_sync_index = phot_sync_idx
#
#     # Save simulated data to netCDF
#     sim_data = xr.Dataset(
#         data_vars=dict(
#             time_tag=(['time_tag_index'], time_tag),
#             time_tag_sync_index=(['time_tag_index'], time_tag_sync_index),
#             true_time_tag=(['true_time_tag_index'], true_time_tag),
#             true_time_tag_sync_index=(['true_time_tag_index'], true_time_tag_sync_index),
#             laser_pulse_width=laser_pulse_width,
#             target_time=target_time,
#             target_amplitude=target_amplitude,
#             background=background
#         ),
#         coords=dict(
#             sync_index=(['sync_index'], sync_index)
#         )
#     )
#
#     save_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\SPCM\Data\Simulated'
#     fname = r'\sim_amp{:.1E}_nshot{:.1E}.nc'.format(target_amplitude, Nshot)
#     sim_data.to_netcdf(save_dir+fname)
#
#     flight_time, n_shots, t_det_lst_ref = dorg.data_organize(dt_sim, save_dir, fname, window_bnd,
#                                                                      max_lsr_num=1e2,
#                                                                      exclude_shots=False)
#
#     # Scaled time-of-flight histogram
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     bin_array = set_binwidth(window_bnd[0], window_bnd[1], dt_sim)
#     n, bins = np.histogram(flight_time, bins=bin_array)
#     binwidth = np.diff(bins)[0]
#     N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
#     center = 0.5 * (bins[:-1] + bins[1:])
#     ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5, label='detected photons')
#     # n, bins = np.histogram(true_flight_time, bins=bin_array)
#     # binwidth = np.diff(bins)[0]
#     # N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
#     # center = 0.5 * (bins[:-1] + bins[1:])
#     # ax.bar(center, N, align='center', width=binwidth, color='r', alpha=0.5, label='true photons')
#     ax.set_title('Arrival Rate Histogram')
#     ax.set_xlabel('time [s]')
#     ax.set_ylabel('Photon Arrival Rate [Hz]')
#     plt.legend()
#     plt.show()

