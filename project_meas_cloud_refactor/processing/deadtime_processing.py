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

    def optimize_complexity(self, t_binedges, r_binedges, cnts_train, cnts_val, low_gain, degree_start, degree_end):
        (
            cnts_1D_train,
            af_hist_1D_train,
            r_centers_trim_t,
            Nshots_train,
            r_binsize_t,
            r_centers_trim
        ) = self.condition_fitting(
            t_binedges,
            r_binedges,
            cnts_train,
            low_gain
        )

        (
            cnts_1D_val,
            af_hist_1D_val,
            __,
            Nshots_val,
            __,
            __
        ) = self.condition_fitting(
            t_binedges,
            r_binedges,
            cnts_val,
            low_gain
        )

        degrees = np.arange(degree_start, degree_end + 1)  # polynomial orders to iterate over
        loss_list_tot_dead = []
        loss_list_tot_pois = []
        loss_train_dead = []
        loss_train_pois = []
        lamb_out_dead_tot = []
        lamb_out_pois_tot = []
        loss_val_dead_tot = []
        loss_val_pois_tot = []
        for degree in degrees:
            results = self.deadtime_fitting(
                degree,
                r_centers_trim_t,
                cnts_1D_train,
                cnts_1D_val,
                af_hist_1D_train,
                af_hist_1D_val,
                Nshots_train,
                Nshots_val
            )
            lamb_out_dead, model_C_dead, model_B_dead, loss_list_dead, loss_val_dead = results['deadtime']
            lamb_out_pois, model_C_pois, model_B_pois, loss_list_pois, loss_val_pois = results['poisson']

            loss_train_dead.append(loss_list_dead[-1])
            loss_train_pois.append(loss_list_pois[-1])
            loss_list_tot_dead.append(loss_list_dead)
            loss_list_tot_pois.append(loss_list_pois)
            lamb_out_dead_tot.append(lamb_out_dead)
            lamb_out_pois_tot.append(lamb_out_pois)

            loss_val_dead_tot.append(loss_val_dead)
            loss_val_pois_tot.append(loss_val_pois)

        min_loss_idx_dead = np.argmin(loss_val_dead_tot)
        min_loss_idx_pois = np.argmin(loss_val_pois_tot)
        optimal_degree_dead = degrees[min_loss_idx_dead]
        optimal_degree_pois = degrees[min_loss_idx_pois]
        lamb_out_dead_best = lamb_out_dead_tot[min_loss_idx_dead]
        lamb_out_pois_best = lamb_out_pois_tot[min_loss_idx_pois]
        loss_list_dead_best = loss_list_tot_dead[min_loss_idx_dead]
        loss_list_pois_best = loss_list_tot_pois[min_loss_idx_pois]

        # print('Deadtime loss vals: {}'.format(loss_train_dead))
        # print('Poisson loss vals: {}'.format(loss_train_pois))
        print('Optimal degrees: Poisson {}, Deadtime {}'.format(optimal_degree_pois, optimal_degree_dead))
        plot_fits(
            cnts_1D_train,
            cnts_1D_val,
            r_binsize_t,
            Nshots_train,
            r_centers_trim,
            lamb_out_pois_best,
            lamb_out_dead_best,
            optimal_degree_pois,
            optimal_degree_dead,
            loss_list_pois_best,
            loss_list_dead_best
        )

    def condition_fitting(self, t_binedges, r_binedges, cnts, low_gain):
        af_hist, deadtime_trim_idx = self.gen_active_fraction(t_binedges, r_binedges, cnts, low_gain)
        cnts_trim = cnts[deadtime_trim_idx:, :]  # Trim count histogram to match active-fraction histogram

        num_col = np.sum(~np.isnan(cnts_trim).all(axis=0))
        cnts_1D = torch.from_numpy(np.nansum(cnts_trim, axis=1)).float()
        af_hist_1D = torch.from_numpy(np.nansum(af_hist, axis=1)).float() / num_col
        r_binedges = torch.from_numpy(r_binedges).float()

        rep_rate = 14.3e3  # [Hz]
        # t_range = t_binedges[-1] - t_binedges[0]  # [s]
        dr = np.diff(t_binedges)[0]
        Nshots = dr * num_col * rep_rate
        r_binsize = torch.diff(r_binedges)[0]  # [m] range bin size in meters
        r_binsize_t = range_to_time(r_binsize, self.c)  # [s] range bin size in seconds
        r_centers_trim = r_binedges[deadtime_trim_idx:-1] + r_binsize / 2  # trimmed to match active-fraction histogram
        r_centers_trim_t = range_to_time(r_centers_trim, self.c)  # [s] convert range to time for optimization

        return cnts_1D, af_hist_1D, r_centers_trim_t, Nshots, r_binsize_t, r_centers_trim

    def deadtime_fitting(
            self,
            degree,
            r_centers_trim_t,
            cnts_1D_train,
            cnts_1D_val,
            af_hist_1D_train,
            af_hist_1D_val,
            Nshots_train,
            Nshots_val
    ):
        """
        Process the histogram data via parametric fitting using the deadtime noise model
        """
        lr = 1e-1  # Learning rate
        rel_step_lim = 1e-8
        max_epochs = 10000
        term_persist = 20

        results = {}
        for mode in ['deadtime', 'poisson']:
            results[mode] = optimize(
                t=r_centers_trim_t,
                Y_train=cnts_1D_train,
                Y_val=cnts_1D_val,
                Z_train=af_hist_1D_train,
                Z_val=af_hist_1D_val,
                Nshots_train=Nshots_train,
                Nshots_val=Nshots_val,
                degree=degree,
                deadtime=(mode == "deadtime"),
                learning_rate=lr,
                rel_step_lim=rel_step_lim,
                max_epochs=max_epochs,
                term_persist=term_persist
            )

        return results

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
