"""
Objective: Script to handle plotting functions
"""

import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import os

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
                         figsize=(self.figsize[0],
                                  self.figsize[1]),
                         constrained_layout=True
                         )
        ax = fig.add_subplot(111)
        ax.scatter(shots_time,
                   ranges / 1e3,
                   s=self.dot_size,
                   alpha=self.alpha,
                   linewidths=0
                   )
        ax.set_ylim(self.ylim) if self.plot_ylim else ax.set_ylim([0, time_to_range(1/self.PRF, self.c) / 1e3])
        ax.set_xlim(self.xlim) if self.plot_xlim else None
        ax.set_xlabel(timestamp.strftime("Time in seconds since %H:%M:%S %Z") if timestamp else 'Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title(
            timestamp.strftime(
                "CoBaLT Backscatter\n{} %Y-%m-%d %H:%M:%S %Z (UTC%z)".format("Low Gain" if low_gain else "High Gain")
            )
        ) if timestamp else ax.set_title('Simulated Backscatter')
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
            ax.plot(flux, r_centers, '-')
            ax.set_xlabel('Flux [Hz]')
            ax.set_ylabel('Range [m]')
            ax.set_title('Flux Histogram')
            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi,
                             figsize=self.figsize
                             )
            ax = fig.add_subplot(111)
            mesh = ax.pcolormesh(t_binedges,
                                 r_binedges / 1e3,
                                 flux,
                                 cmap='viridis',
                                 norm=LogNorm(vmin=flux[flux > 0].min(),
                                              vmax=flux.max())
                                 # norm=LogNorm(2e5,
                                 #              2e9)
                                 )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
            ax.set_xlabel(timestamp.strftime("Time in seconds since %H:%M:%S %Z") if timestamp else 'Time [s]')
            ax.set_ylabel('Range [km]')
            ax.set_title(
                timestamp.strftime(
                    "CoBaLT Backscatter\n{} %Y-%m-%d %H:%M:%S %Z (UTC%z)\n{:.2e} m x {:.2e} s".format(
                        "Low Gain" if low_gain else "High Gain", dr, dt)
                )
            ) if timestamp else ax.set_title('Simulated Backscatter\n{:.2e} m x {:.2e} s'.format(dr, dt))
            ax.set_ylim(self.ylim) if self.plot_ylim else ax.set_ylim([0, time_to_range(1/self.PRF, self.c) / 1e3])
            ax.set_xlim(self.xlim) if self.plot_xlim else None
            plt.tight_layout()
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
            cnts_1D,
            r_binsize_t,
            Nshots,
            r_centers_trim,
            lamb_out_pois,
            lamb_out_dead,
            degree,
            loss_list_pois,
            loss_list_dead
    ):
    """
    Plot fit outputs from Optimizer routine and loss behavior during descent.
    """
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(cnts_1D / r_binsize_t / Nshots / 1e6, r_centers_trim / 1e3, 'o', alpha=0.5, label='Raw')
    ax.plot(lamb_out_pois / 1e6, r_centers_trim / 1e3, '-', alpha=0.7, label='Poisson Fit')
    ax.plot(lamb_out_dead / 1e6, r_centers_trim / 1e3, '-', alpha=0.7, label='Deadtime Fit')
    # if use_sim:
    #     ax.plot(lamb / 1e6, r / 1e3, '-', alpha=0.7, label='Simulated Truth')
    #     ax.set_ylim([loader.r_plot_min, loader.r_plot_max])
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

def plot_af_histogram(t_binedges, tbinsize, r_binedges, deadtime_trim_idx, af_hist):
    """
    Plot active-fraction histogram.
    """
    extent_t0, extent_t1 = (
        (t_binedges[0] - (tbinsize / 2)),
        (t_binedges[-1] + (tbinsize / 2))
    )  # [s, s]
    extent_r0, extent_r1 = (
        (r_binedges[deadtime_trim_idx] / 1e3),
        (r_binedges[-1] / 1e3)
    )  # [km, km]

    fig = plt.figure(
        figsize=(6, 4),
        dpi=400
    )
    ax = fig.add_subplot(111)
    im = ax.imshow(
        af_hist,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[
            extent_t0,
            extent_t1,
            extent_r0,
            extent_r1
        ]
    )
    im.set_clim(0, 1)
    cbar = fig.colorbar(
        im,
        ax=ax
    )
    cbar.set_label('AF Value')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Range [km]')
    plt.show()
