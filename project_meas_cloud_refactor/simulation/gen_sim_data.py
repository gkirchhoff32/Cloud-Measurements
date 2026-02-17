"""
Objective: Generate simulated data and save to netCDF (.nc) file
"""

import numpy as np
import xarray as xr
import time
from pathlib import Path

from physics.math import gaussian
from simulation import sim_deadtime_utils as sim

class GenerateSimData:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System params
        self.deadtime = config['system_params']['deadtime_hg']  # [s] high-gain detector deadtime
        self.laser_pulse_width = config['sim_params']['laser_pulse_width']  # [s] laser pulse width

        # Simulation params
        self.r_sim_min, self.r_sim_max = config['sim_params']['sim_ylim']  # [km] range window
        self.Nshot = int(config['sim_params']['Nshot'])  # number of laser shots
        self.wrap_deadtime = False

        # Plotting params
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size

        # Gaussian params
        self.mu = config['sim_params']['mu']  # [m] Gaussian center
        self.sigma = config['sim_params']['sigma']  # [m] Gaussian stdev
        self.A = config['sim_params']['A']  # [Hz] Gaussian amplitude flux
        self.b = config['sim_params']['b']  # [Hz] background flux

        # Save params
        self.save_loc = config['file_params']['save_dir']
        self.func_shape = config['file_params']['func_shape']

    def write_sim_data(self):
        # TODO: CLEAN THIS UP! and continue simulated data loader
        r = np.arange(self.r_sim_min * 1e3, self.r_sim_max * 1e3 + self.rbinsize, self.rbinsize)  # [m] range axis

        r_t = r / self.c * 2  # [s] range axis in time
        mu_t = self.mu / self.c * 2  # [s] center of guassian in time
        sigma_t = self.sigma / self.c * 2  # [s] spread of gaussian in time
        dr_t = self.rbinsize / self.c * 2  # [s] range bin resolution in time

        r_t_min = r_t[0]  # [s] beginning of range window to shift
        r_t_shifted = r_t - r_t_min  # [s] shift the time axis
        mu_t_shifted = mu_t - r_t_min  # [s] shift the center of the Gaussian
        if self.func_shape == 'gaussian':
            lamb = gaussian(r_t_shifted, self.A, mu_t_shifted, sigma_t, self.b)
        else:
            print('Make sure to select appropriate function from physics.math.py module.')
            raise ValueError

        start = time.time()
        # Generate simulated data
        sim_results = self.gen_sim_data(
            photon_rate_arr=lamb,
            t_sim_bins=r_t_shifted,
            tD=self.deadtime,
            Nshot=self.Nshot,
            wrap_deadtime=self.wrap_deadtime,
        )
        print('Simulated Data generated. Time elapsed: {:.1f} s'.format(time.time() - start))

        sync_idx = sim_results['sync_idx']  # laser sync events
        time_tag = sim_results['det_events']  # detection time tags
        true_time_tag = sim_results['phot_events']  # incident photon time tags
        time_tag_sync_idx = sim_results['det_sync_idx']  # sync index for detections
        true_time_tag_sync_idx = sim_results['phot_sync_idx']  # sync index for incident photons
        t_sim_bins = sim_results['t_sim_bins']

        dr_t = np.diff(t_sim_bins)[0]  # [s]
        time_tag += r_t_min / dr_t  # shift back to actual time values (in units clock counts)
        true_time_tag += r_t_min / dr_t  # shift back to actual time values (in units clock counts)

        # Save simulated data to netCDF
        sim_data = xr.Dataset(data_vars=dict(
            time_tag=(['time_tag_index'], time_tag),
            time_tag_sync_index=(['time_tag_index'], time_tag_sync_idx),
            true_time_tag=(['true_time_tag_index'], true_time_tag),
            true_time_tag_sync_index=(['true_time_tag_index'], true_time_tag_sync_idx),
            laser_pulse_width=self.laser_pulse_width,
            target_time=mu_t,
            target_sigma=sigma_t,
            target_amplitude=self.A,
            background=self.b,
            dt_sim=dr_t,
            time_axis=r_t,
            profile=self.func_shape
        ),
            coords=dict(
                sync_index=(['sync_index'], sync_idx)
            )
        )

        fname = r'\sim_{}_A{:.1E}Hz_mu{:.1f}km_sig{:.1E}m_N{}.nc'.format(
            self.func_shape,
            self.A,
            self.mu/1e3,
            self.sigma,
            self.Nshot
        )
        home = str(Path.home())
        save_dir = home + self.save_loc + self.func_shape
        sim_data.to_netcdf(save_dir + fname)

        return lamb, r

    @staticmethod
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
        t_sim_bins = np.concatenate((t_sim_bins, t_sim_bins[-1:] + dt_sim))  # simulation time histogram bins

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
            ptime, ctime = sim.photon_count_generator(
                t_sim_bins,
                photon_rate_arr,
                tau_d_flt=tD,
                last_photon_flt=t_det_last
            )
            if wrap_deadtime:
                if len(ctime) > 0:
                    t_det_last = ctime[-1]
                t_det_last -= t_sim_bins[-1]

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