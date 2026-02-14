"""
Objective: Generate histogram for processing or visualization
"""

import numpy as np
import time

from physics.conversions import time_to_range, range_to_time, convert_flux

class GenerateHistogram:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System params
        self.deadtime_hg = config['system_params']['deadtime_hg']  # [s]
        self.deadtime_lg = config['system_params']['deadtime_lg']  # [s]
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Plot params
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.tbinsize = config['plot_params']['tbinsize']  # [s] temporal bin size
        self.ylim = config['plot_params']['ylim']  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits

        # Process params
        self.load_xlim = config['process_params']['load_xlim']  # TRUE value limits range when generating histogram
        self.load_ylim = config['process_params']['load_ylim']  # TRUE value limits range when generating histogram
        self.active_fraction = config['process_params']['active_fraction']

    def gen_histogram(self, ranges, shots_time, low_gain):
        # Define scales for histogram generation
        rbinsize, tbinsize = self.calc_binsize()

        # Determine binedges
        ranges, shots_time, rbins, tbins, reduce_min = self.calc_bins(ranges, low_gain, rbinsize, shots_time, tbinsize)

        # Generate histogram
        flux, H, t_binedges, r_binedges = self.build_histogram(shots_time, ranges, tbinsize, rbinsize, tbins, rbins)

        return r_binedges, t_binedges, flux, H

    def calc_binsize(self):
        """
        Calculate bin sizes for histogram
        """
        dr_af = self.rbinsize  # [m]
        dt_af = 1 / self.PRF  # [s]

        # Round time histogram bin size that factorizes the fine-res bin size
        t_factor = max(1, round(self.tbinsize / dt_af))
        tbinsize_close = t_factor * dt_af  # [s]
        rbinsize = dr_af  # [m]
        tbinsize = tbinsize_close

        return rbinsize, tbinsize

    def calc_bins(self, ranges, low_gain, rbinsize, shots_time, tbinsize):
        # Set time window
        if self.load_xlim:
            min_time, max_time = self.xlim[0], self.xlim[1]  # [s]
            max_shots_idx = np.argmin(np.abs(shots_time - max_time))
            min_shots_idx = np.argmin(np.abs(shots_time - min_time))
            shots_time = shots_time[min_shots_idx:max_shots_idx]
            ranges = ranges[min_shots_idx:max_shots_idx]

        # deadtime calculations
        deadtime = self.deadtime_lg if low_gain else self.deadtime_hg
        deadtime_range = time_to_range(deadtime, self.c)  # [m]

        # Set range window. If calculating active fraction, then load data preceding ymin by one deadtime interval
        if self.load_ylim:
            reduce_min = deadtime_range if self.active_fraction and (deadtime_range >= rbinsize) else 0
            min_range, max_range = (self.ylim[0] * 1e3 - reduce_min), (self.ylim[1] * 1e3)  # [m]
        else:
            reduce_min = 0
            max_range_time = 1 / self.PRF  # [s]
            min_range, max_range = 0, time_to_range(max_range_time, self.c)  # [m]

        min_time, max_time = shots_time[0], shots_time[-1]  # [s]
        rbins = np.arange(min_range, max_range + rbinsize, rbinsize)  # [m]
        tbins = np.arange(min_time, max_time, tbinsize)  # [s]

        print('Actual range and time bin sizes: {:.3e} m x {:.3e} s'.format(rbinsize, tbinsize))

        return ranges, shots_time, rbins, tbins, reduce_min

    def build_histogram(self, shots_time, ranges, tbinsize, rbinsize, tbins, rbins):
        start = time.time()
        # Generate histogram of counts
        H, t_binedges, r_binedges = np.histogram2d(shots_time, ranges, bins=[tbins, rbins])  # Generate 2D histogram
        H = H.T  # flip axes

        # Calculate flux from counts
        rbinsize_time = range_to_time(rbinsize, self.c)  # [s] range bin size in time
        N = tbinsize * self.PRF  # number of integrated shots per time bin
        flux = convert_flux(H, rbinsize_time, N)  # [Hz] Backscatter flux calculation

        print('Finished generating histogram.\nTime elapsed: {:.1f} s'.format(time.time() - start))

        return flux, H, t_binedges, r_binedges



