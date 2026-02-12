import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import sys
import torch

project_root = Path(__file__).resolve().parent.parent
utils_dir = project_root / "processing_methods"
sys.path.append(str(utils_dir))

from data_optimizer_utils import optimize

class DataProcessor:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

    def gen_histogram_processing(self, use_sim, loader, dpp):
        if use_sim:
            lamb, r = loader.generate_data()
            histogram_results = loader.load_sim_data()
        else:
            dpp.run(use_sim)
            lamb, r = None, None
            histogram_results = loader.gen_histogram()

        return {
            'histogram_results': histogram_results,
            'lamb': lamb,
            'r': r
        }

    def generate_fits(self, use_sim, loader, dpp, histogram_processing_results):
        histogram_results = histogram_processing_results['histogram_results']
        lamb = histogram_processing_results['lamb']
        r = histogram_processing_results['r']

        flux_vars = self.data_setup(loader, dpp.deadtime_correct, histogram_results)

        cnts_raw_fine = flux_vars['cnts_raw']
        af_hist_fine = flux_vars['af_hist']
        t_binedges_fine = flux_vars['t_binedges']
        r_binedges_fine = flux_vars['r_binedges']

        num_tbins = af_hist_fine.shape[1]
        cnts_raw = torch.from_numpy(cnts_raw_fine.sum(axis=1)).float()
        af_hist = torch.from_numpy(af_hist_fine.sum(axis=1)).float() / num_tbins
        r_binedges = torch.from_numpy(r_binedges_fine).float()

        # flux_raw_fine = cnts_raw_fine / np.diff(t_binedges_fine)[0] / 14.3e3 / (np.diff(r_binedges_fine)[0] / c * 2)

        rep_rate = 14.3e3  # [Hz]
        t_range = t_binedges_fine[-1] - t_binedges_fine[0]  # [s]
        Nshots = t_range * rep_rate
        r_binsize = torch.diff(r_binedges)[0]  # [m] range bin size in meters
        r_binsize_t = self.range_to_time(r_binsize)  # [s] range bin size in seconds
        r_centers = r_binedges[:-1] + r_binsize / 2
        r_centers_t = self.range_to_time(r_centers)  # [s] convert range to time for optimization

        degree = 20
        num_steps = 2000
        lr = 1e-1  # Learning rate
        rel_step_lim = 1e-8
        max_epochs = 10000
        term_persist = 20

        results = {}
        for mode in ['deadtime', 'poisson']:
            results[mode] = optimize(
                Y=cnts_raw,
                Z=af_hist,
                t=r_centers_t,
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

        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111)
        ax.plot(cnts_raw / r_binsize_t / Nshots / 1e6, r_centers / 1e3, 'o', alpha=0.5, label='Raw')
        ax.plot(lamb_out_pois / 1e6, r_centers / 1e3, '-', alpha=0.7, label='Poisson Fit')
        ax.plot(lamb_out_dead / 1e6, r_centers / 1e3, '-', alpha=0.7, label='Deadtime Fit')
        if use_sim:
            ax.plot(lamb / 1e6, r / 1e3, '-', alpha=0.7, label='Simulated Truth')
            ax.set_ylim([loader.r_plot_min, loader.r_plot_max])
        ax.set_xlabel('Flux [MHz]')
        ax.set_ylabel('Range [km]')
        ax.set_title('Fit: Degree {}'.format(degree))
        # ax.set_xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111)
        ax.plot(range(len(loss_list_pois)), loss_list_pois, label='Poisson')
        ax.plot(range(len(loss_list_dead)), loss_list_dead, label='Deadtime')
        ax.set_title('Loss Values')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def range_to_time(self, r):
        return r / self.c * 2  # [s]

    def data_setup(self, loader, deadtime_correct, histogram_results):
        flux_raw = histogram_results['flux_raw']  # [Hz]
        cnts_raw = histogram_results['cnts_raw']
        af_results = deadtime_correct.calc_af_hist_convolution(histogram_results, loader)
        af_hist = af_results['af_hist']

        deadtime_trim_idx = deadtime_correct.deadtime_trim_idx
        flux_raw = flux_raw[deadtime_trim_idx:, :]  # [Hz] Removing initial loaded bins for AF hist calculation
        cnts_raw = cnts_raw[deadtime_trim_idx:, :]  # Removing initial loaded bins for AF hist calculation
        t_binedges = histogram_results['t_binedges']
        r_binedges = histogram_results['r_binedges'][deadtime_trim_idx:]

        flux_bin_est = flux_raw / af_hist

        self.plot_flux_est(flux_raw, flux_bin_est, t_binedges, r_binedges)

        return {'flux_raw': flux_raw,
                'cnts_raw': cnts_raw,
                'af_hist': af_hist,
                't_binedges': t_binedges,
                'r_binedges': r_binedges
                }

    def plot_flux_est(self, flux_raw, flux_est, t_binedges, r_binedges):
        vmin = np.nanmin(flux_raw[flux_raw > 0]) / 1e6
        # mask_inf_dc = np.isfinite(flux_est) & (flux_est <= 40e16)  # mask to remove infinite values and anything too large
        mask_inf_dc = np.isfinite(flux_est)  # mask to remove infinite values and anything too large
        vmax = np.nanmax(flux_est[mask_inf_dc]) / 1e6

        fig = plt.figure(dpi=400,
                         figsize=(8, 6),
                         constrained_layout=True
                         )
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        __ = ax1.pcolormesh(t_binedges,
                            r_binedges / 1e3,
                            flux_raw / 1e6,
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin,
                                         vmax=vmax
                                         )
                            )
        mesh2 = ax2.pcolormesh(t_binedges,
                               r_binedges / 1e3,
                               flux_est / 1e6,
                               cmap='viridis',
                               norm=LogNorm(vmin=vmin,
                                            vmax=vmax
                                            )
                               )
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Range [km]')
        ax1.set_title('Raw')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Bin Correction')
        ax2.tick_params(labelleft=False)
        cbar = fig.colorbar(mesh2, ax=[ax1, ax2],
                            location='right',
                            pad=0.15)
        cbar.set_label('Flux [MHz]')
        [plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') for ax in [ax1, ax2]]
        plt.show()

    def bin_corrections_process(self, loader, plotter, deadtime_correct):
        """
        Process data: Generate histogram --> Mueller correction --> deadtime-model correction --> background correction
        """
        histogram_results = loader.gen_histogram()

        # Calculate Mueller correction
        mueller_results = deadtime_correct.mueller_correct(histogram_results, loader)

        # Calculate deadtime-model correction
        af_results = deadtime_correct.calc_af_hist_convolution(histogram_results, loader)
        dc_results = deadtime_correct.deadtime_model_correct(af_results, histogram_results)
        deadtime_bg_results = deadtime_correct.deadtime_bg_calc(loader, plotter)

        # Compare corrections
        fluxes_bg_sub = deadtime_correct.plot_binwise_corrections(mueller_results, dc_results,
                                                                  deadtime_bg_results, loader)

        return fluxes_bg_sub
    
    def repeat_process(self, loader, plotter, deadtime_correct, num_seq):
        """
        Placeholder for data processing methods without corrections.
        """
        for i in range(num_seq):

            self.corrections_process(loader, plotter, deadtime_correct)
