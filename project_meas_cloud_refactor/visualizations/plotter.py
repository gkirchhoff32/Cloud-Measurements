"""
Objective: Script to handle plotting functions
"""

import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import LogLocator, FuncFormatter
from pathlib import Path
import os
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.path_utils import get_unique_filename, find_data_path
from physics.conversions import time_to_range

class DataPlotter:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Plot params
        self.dpi = config['plot_params']['dpi']  # dots-per-inch
        self.figsize = config['plot_params']['figsize']  # figure size in inches
        self.dot_size = config['plot_params']['dot_size']  # Dot size for 'axes.scatter' 's' param
        self.alpha = config['plot_params']['alpha']  # alpha value when plotting
        self.plot_xlim = config['plot_params']['plot_xlim']  # TRUE value limits range only when plotting
        self.plot_ylim = config['plot_params']['plot_ylim']  # TRUE value limits range only when plotting
        self.ylim = config['plot_params']['ylim']  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits
        self.save_img = config['plot_params']['save_img']  # Save images if TRUE
        self.save_dpi = config['plot_params']['save_dpi']  # DPI for saved images

        # File params
        self.date = config['file_params']['date']  # Date directory
        self.image_dir = config['file_params']['image_dir']  # Directory to save images
        self.data_dir = config['file_params']['data_dir_win'] if os.name == 'nt' else config['file_params']['data_dir_lin']

    def plot_time_tag_scatter(self, ranges, shots_time, timestamp, low_gain, generic_fname):
        # Start plotting
        print('\nStarting to generate scatter plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi,
                         figsize=self.figsize
                         )
        # gs = GridSpec(1, 20, figure=fig)
        # ax = fig.add_subplot(gs[0, :18])
        ax = fig.add_subplot(111)
        ax.scatter(shots_time,
                   ranges / 1e3,
                   s=self.dot_size,
                   alpha=self.alpha,
                   linewidths=0
                   )
        ax.set_ylim(self.ylim) if self.plot_ylim else ax.set_ylim([0, time_to_range(1 / self.PRF, self.c) / 1e3])
        ax.set_xlim(self.xlim) if self.plot_xlim else None

        # ax.set_box_aspect(1)  # makes the actual plotting area square

        ax.set_xlabel(timestamp.strftime("Time in seconds since %H:%M:%S %Z") if timestamp else 'Time [s]')
        ax.set_ylabel('Range [km]')
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')
        # ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
        ax.set_title(
            timestamp.strftime(
                "CoBaLT Backscatter\n{} %Y-%m-%d %H:%M:%S %Z (UTC%z)".format("Low Gain" if low_gain else "High Gain")
            ),
            pad=20
        ) if timestamp else ax.set_title('Simulated Backscatter', pad=20)

        fig.subplots_adjust(
            left=0.22,
            bottom=0.2
        )

        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        if self.save_img:
            print('Starting to save image...')
            start = time.time()
            img_fname = generic_fname + '_scatter' + '.png'
            self.data_dir = find_data_path(self.data_dir)
            img_save_path = Path(self.data_dir + self.image_dir + self.date) / img_fname
            fname = get_unique_filename(img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
            print('Finished saving plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        plt.show()

    def plot_histogram(self, flux, t_binedges, r_binedges, timestamp, low_gain, generic_fname):
        """
                Plot histogram using results from "gen_histogram" method

                Inputs:
                    histogram_results: output dictionary from "gen_histogram" function
                """
        # Processed data
        dt = t_binedges[1] - t_binedges[0]  # [s]
        dr = r_binedges[1] - r_binedges[0]  # [m]

        # Start plotting
        print('\nStarting to generate histogram plot...')
        start = time.time()

        # plot line graph if histogram is 1D. Heatmap if 2D.
        if flux.shape[1] == 1:
            r_centers = r_binedges[:-1] + dr / 2

            fig = plt.figure(dpi=self.dpi,
                             figsize=self.figsize
                             )
            ax = fig.add_subplot(111)
            ax.plot(flux/1e6, r_centers/1e3, 'o')
            ax.set_xlabel('Flux [MHz]')
            ax.set_ylabel('Range [km]')
            ax.set_title('Flux Histogram')
            # ax.set_ylim([1.8, 1.82])
            # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi,
                             figsize=self.figsize
                             )
            # gs = GridSpec(1, 20, figure=fig)
            # ax = fig.add_subplot(gs[0, :14])
            # cax = fig.add_subplot(gs[0, 16:17])
            ax = fig.add_subplot(111)
            mesh = ax.pcolormesh(t_binedges,
                                 r_binedges / 1e3,
                                 flux/1e6,
                                 cmap='viridis',
                                 # norm=LogNorm(vmin=flux[flux > 0].min()/1e6,
                                 #              vmax=flux.max()/1e6)
                                 norm=LogNorm(2e-1,
                                              9e2)
                                 )
            cax = inset_axes(
                ax,
                width="4%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.04, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )

            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label('Flux [MHz]')
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label('Flux [MHz]')
            # ticks = [0.6, 1, 2]
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels([str(t) for t in ticks])
            # ticks = [200, 300, 400, 600, 1000, 2000]
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels([f"{t:g}" for t in ticks])
            ax.set_xlabel(timestamp.strftime("Time in seconds since %H:%M:%S %Z") if timestamp else 'Time [s]')
            ax.set_ylabel('Range [km]')
            ax.set_title(
                timestamp.strftime(
                    "CoBaLT Backscatter\n{} %Y-%m-%d %H:%M:%S %Z (UTC%z)\n{:.2e} m x {:.2e} s".format(
                        "Low Gain" if low_gain else "High Gain", dr, dt)
                ),
                pad=20
            ) if timestamp else ax.set_title('Simulated Backscatter\n{:.2e} m x {:.2e} s'.format(dr, dt))
            ax.set_ylim(self.ylim) if self.plot_ylim else ax.set_ylim([0, time_to_range(1 / self.PRF, self.c) / 1e3])
            ax.set_xlim(self.xlim) if self.plot_xlim else None

            ax.set_box_aspect(1)  # makes the actual histogram panel square

            # ax.set_xlim([160, 215])
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            # fig.subplots_adjust(left=0.25, bottom=0.18, right=1, top=0.92)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            # plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
            # plt.tight_layout()

            fig.subplots_adjust(
                left=0.18,
                right=0.8,
            )

            print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
            if self.save_img:
                print('Starting to save image...')
                start = time.time()
                img_fname = generic_fname + '_hg' + '.png'
                self.data_dir = find_data_path(self.data_dir)
                img_save_path = Path(self.data_dir + self.image_dir + self.date) / img_fname
                fname = get_unique_filename(img_save_path)
                fig.savefig(fname, dpi=self.save_dpi)
                print('Finished saving plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
            plt.show()

def plot_fits(
        cnts_1D_train,
        cnts_1D_val,
        r_binsize_t,
        Nshots,
        r_centers_trim,
        lamb_out_pois,
        lamb_out_dead,
        degree_pois,
        degree_dead,
        loss_list_pois,
        loss_list_dead
    ):
    """
    Plot fit outputs from Optimizer routine and loss behavior during descent.
    """
    avg_flux_train = torch.mean(cnts_1D_train/r_binsize_t/Nshots)  # [Hz]
    avg_flux_val = torch.mean(cnts_1D_val/r_binsize_t/Nshots)  # [Hz]
    avg_flux = (avg_flux_train + avg_flux_val) / 2  # [Hz]
    avg_flux_10x10 = avg_flux * (1.2 / 10) * (0.1 / 10)  # [Hz]
    avg_flux_10x10_mueller = avg_flux_10x10 / (1 - 31.8e-9 * avg_flux_10x10)  # [Hz]
    print('Coarse (10 m x 10 s) flux estimate: {:.2f} kHz'.format(avg_flux_10x10_mueller/1e3))
    # avg_flux_corrected = avg_flux / (1 - 29.5e-9 * avg_flux)  # [Hz]

    cnts_1D = torch.cat((cnts_1D_train, cnts_1D_val), dim=0)
    r_centers_trim_cat = torch.cat((r_centers_trim, r_centers_trim), dim=0)

    fig = plt.figure(
        dpi=400,
        figsize=(4, 4)
    )
    ax = fig.add_subplot(111)
    ax.plot(cnts_1D/r_binsize_t/Nshots/1e6, r_centers_trim_cat/1e3, '.', color='#4A4A4A', markeredgewidth=0, alpha=0.35, label='Raw')
    # ax.plot(cnts_1D_train/r_binsize_t/Nshots/1e6, r_centers_trim/1e3, '.', color="red", markeredgewidth=0, alpha=0.25, label='Raw (train)')
    # ax.plot(cnts_1D_val/r_binsize_t/Nshots/1e6, r_centers_trim/1e3, 's', color="#4A4A4A", markersize=3, mec=None, alpha=0.25, label='Raw (validation)')
    # ax.plot(lamb_out_pois / 1e6, r_centers_trim / 1e3, '-', color="#000000", alpha=0.8, label='Poisson Fit')
    ax.plot(lamb_out_dead / 1e6, r_centers_trim / 1e3, '-', color="#1B4F72", alpha=0.8, label='Estimate')
    ax.axvline(
        x=avg_flux_10x10_mueller/1e6,
        color="#FF69B4",  # hot pink
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label='Coarse'
    )
    ax.set_xlabel('Flux [MHz]')
    ax.set_ylabel('Range [km]')
    ax.set_title('Fit: Poisson Degree {}, Deadtime Degree {}'.format(degree_pois, degree_dead))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=5))
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
    ax.set_yticks(np.array([1.2385, 1.2388, 1.2391, 1.2394, 1.2397]))
    ax.set_ylim([1.2384, 1.2398])
    # ax.set_xlim([0, 300])
    # ax.set_xscale('log')
    plt.legend(fontsize=8)
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

def plot_af_histogram(t_binedges, tbinsize, r_binedges, deadtime_trim_idx, af_hist):
    """
    Plot active-fraction histogram.
    """
    # extent_t0, extent_t1 = (
    #     (t_binedges[0] - (tbinsize / 2)),
    #     (t_binedges[-1] + (tbinsize / 2))
    # )  # [s, s]
    # extent_r0, extent_r1 = (
    #     (r_binedges[deadtime_trim_idx] / 1e3),
    #     (r_binedges[-1] / 1e3)
    # )  # [km, km]

    fig = plt.figure(dpi=400, figsize=(3, 4))
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(t_binedges,
                         r_binedges[deadtime_trim_idx:] / 1e3,
                         af_hist,
                         cmap='viridis',
                         vmin=0,
                         vmax=1
                         )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('AF Value')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Range [km]')
    plt.tight_layout()
    plt.show()

    # fig = plt.figure(
    #     figsize=(6, 4),
    #     dpi=400
    # )
    # ax = fig.add_subplot(111)
    # im = ax.imshow(
    #     af_hist,
    #     aspect='auto',
    #     origin='lower',
    #     cmap='viridis',
    #     extent=[
    #         extent_t0,
    #         extent_t1,
    #         extent_r0,
    #         extent_r1
    #     ]
    # )
    # im.set_clim(0, 1)
    # cbar = fig.colorbar(
    #     im,
    #     ax=ax
    # )
    # cbar.set_label('AF Value')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Range [km]')
    # plt.show()
