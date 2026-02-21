"""
Objective: Process data using deadtime noise model (single bin or full scene).
"""

import time
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import torch

from physics.conversions import time_to_range, range_to_time
from processing.optimizer import optimize
from visualizations.plotter import plot_fits, plot_af_histogram

class DeadtimeProcessing:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System params
        self.deadtime_hg = config['system_params']['deadtime_hg']  # [s] high-gain detector deadtime
        self.deadtime_lg = config['system_params']['deadtime_lg']  # [s] low-gain detector deadtime
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Plot params
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.tbinsize = config['plot_params']['tbinsize']  # [s] time bin size
        self.dpi = config['plot_params']['dpi']  # dots-per-inch
        self.figsize = config['plot_params']['figsize']  # figure size in inches

    def deadtime_fitting(self, cnts, r_binedges, t_binedges, low_gain):
        """
        Process the histogram data via parametric fitting using the deadtime noise model
        """
        af_hist, deadtime_trim_idx = self.gen_active_fraction(t_binedges, r_binedges, cnts, low_gain)
        cnts_trim = cnts[deadtime_trim_idx:, :]  # Trim count histogram to match active-fraction histogram

        num_tbins = af_hist.shape[1]
        cnts_1D = torch.from_numpy(cnts_trim.sum(axis=1)).float()
        af_hist_1D = torch.from_numpy(af_hist.sum(axis=1)).float() / num_tbins
        r_binedges = torch.from_numpy(r_binedges).float()

        rep_rate = 14.3e3  # [Hz]
        t_range = t_binedges[-1] - t_binedges[0]  # [s]
        Nshots = t_range * rep_rate
        r_binsize = torch.diff(r_binedges)[0]  # [m] range bin size in meters
        r_binsize_t = range_to_time(r_binsize, self.c)  # [s] range bin size in seconds
        r_centers_trim = r_binedges[deadtime_trim_idx:-1] + r_binsize / 2  # trimmed to match active-fraction histogram
        r_centers_trim_t = range_to_time(r_centers_trim, self.c)  # [s] convert range to time for optimization

        degree = 8
        num_steps = 2000
        lr = 1e-1  # Learning rate
        rel_step_lim = 1e-8
        max_epochs = 10000
        term_persist = 20

        results = {}
        for mode in ['deadtime', 'poisson']:
            results[mode] = optimize(
                Y=cnts_1D,
                Z=af_hist_1D,
                t=r_centers_trim_t,
                Nshots=Nshots,
                num_steps=num_steps,
                degree=degree,
                deadtime=(mode == "deadtime"),
                learning_rate=lr,
                rel_step_lim=rel_step_lim,
                max_epochs=max_epochs,
                term_persist=term_persist
            )

        lamb_out_dead, model_C_dead, model_B_dead, loss_list_dead = results['deadtime']
        lamb_out_pois, model_C_pois, model_B_pois, loss_list_pois = results['poisson']
        print('Background term: deadtime {:.0f} Hz, poisson {:.0f} Hz'.format(model_B_dead[0], model_B_pois[0]))

        plot_fits(
            cnts_1D,
            r_binsize_t,
            Nshots,
            r_centers_trim,
            lamb_out_pois,
            lamb_out_dead,
            degree,
            loss_list_pois,
            loss_list_dead
        )

    def binwise_correction(self, flux, r_binedges, t_binedges, cnts, low_gain):
        """
        Apply single-bin correction based on deadtime-noise model. This essentially inverts the photon-count histogram
        with the active-fraction histogram.
        """
        af_hist, deadtime_trim_idx = self.gen_active_fraction(t_binedges, r_binedges, cnts, low_gain)

        flux_raw = flux[deadtime_trim_idx:]  # [Hz]
        r_binedges = r_binedges[deadtime_trim_idx:]  # [m]

        flux_est = flux_raw / af_hist

        return flux_est, r_binedges

    def mueller_correction(self, flux, low_gain):
        """
        Apply the Mueller correction to flux histogram.
        """
        deadtime = self.deadtime_lg if low_gain else self.deadtime_hg

        # Calculate flux estimate based on Mueller Correction
        flux_mueller = flux / (1 - deadtime * flux)  # [Hz]

        return flux_mueller

    def gen_active_fraction(self, t_binedges, r_binedges, cnts, low_gain):
        """
        Method to calculate active-fraction histogram using fractional binning.

        Args:
            r_binedges: [m] range bin edges. IMPORTANT: Must include one deadtime interval before minimum range
            cnts: histogram photon counts
            rbinsize: [m] range bin size
            tbinsize: [s] time bin size
            PRF: [Hz] laser pulse rate
        """

        print('Calculating active fraction...')

        start_time = time.time()

        deadtime = self.deadtime_lg if low_gain else self.deadtime_hg

        # Generate deadtime kernel for active-fraction calculation
        dtime_kern, deadtime_trim_idx = self.calc_deadtime_kernel(deadtime, self.rbinsize)

        # Calculate active-fraction histogram
        nrbins = len(r_binedges[:-1])
        af_hist = self.calc_active_fraction(self.tbinsize, self.PRF, cnts, dtime_kern, nrbins, deadtime_trim_idx)

        print('Active fraction calculation. Elapsed time: {} s'.format(time.time() - start_time))

        plot_af_histogram(t_binedges, self.tbinsize, r_binedges, deadtime_trim_idx, af_hist)

        return af_hist, deadtime_trim_idx

    def calc_deadtime_kernel(self, deadtime, rbinsize):
        """
        Generate deadtime kernel that will be used to produce active-fraction histogram via convolution.\
        Example kernel: (1, 1, 1, 0.75) --> 3 full bins occupied plus a fractional one.
        """
        deadtime_range = time_to_range(deadtime, self.c)  # [m]

        # Calculate number of bins (floor round) that occupy deadtime
        deadtime_trim_idx = np.floor(deadtime_range / rbinsize).astype(int)
        dtime_kern = np.ones(deadtime_trim_idx)

        # Remainder. If deadtime is longer than binsize, then tack on a fractional bin to the kernel.
        # If deadtime is shorter, then the kernel is solely the fractional bin.
        fractional_bin = deadtime_range % rbinsize if (deadtime_range > rbinsize) else deadtime_range / rbinsize
        dtime_kern = np.append(dtime_kern, fractional_bin)

        return dtime_kern, deadtime_trim_idx

    def calc_active_fraction(self, tbinsize, PRF, cnts, dtime_kern, nrbins, deadtime_trim_idx):
        """
        Calculate active-fraction histogram by convolving count histogram with deadtime kernel.
        """
        N = tbinsize * PRF  # number of shots per time bin
        d = fftconvolve(cnts, dtime_kern[:, None], mode='full') / N
        d = d[:nrbins, :]
        a = 1 - d

        # Trim AF histogram to remove preceding range bins
        af_hist = a[deadtime_trim_idx:, :]

        return af_hist
