import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path


class DataPlotter:
    def __init__(self, config):
        self.flux_bg_sub = False
        self.cbar_min = None
        self.cbar_max = None

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate

        # Plot params
        self.chunksize = 50_000_000  # reasonable value to produce ~700 MB size .nc files
        self.plot_xlim = config['plot_params']['plot_xlim']  # TRUE value limits range only when plotting
        self.plot_ylim = config['plot_params']['plot_ylim']  # TRUE value limits range only when plotting
        self.tbinsize = config['plot_params']['tbinsize']  # [s] temporal bin size
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.dpi = config['plot_params']['dpi']  # dots-per-inch
        self.figsize = config['plot_params']['figsize']  # figure size in inches
        self.ylim = config['plot_params']['ylim']  # [km] y-axis limits
        self.xlim = config['plot_params']['xlim']  # [s] x-axis limits
        self.histogram = config['plot_params']['histogram']  # Plot histogram if TRUE, else scatter plot
        self.save_img = config['plot_params']['save_img']  # Save images if TRUE
        self.save_dpi = config['plot_params']['save_dpi']  # DPI for saved images
        self.dot_size = config['plot_params']['dot_size']  # Dot size for 'axes.scatter' 's' param
        # subtracted and range corrected
        self.chunk_start = config['plot_params']['chunk_start']  # Chunk to start plotting from
        self.chunk_num = config['plot_params']['chunk_num']  # Number of chunks to plot. If exceeds remaining chunks,
        # then it will plot the available ones
        self.alpha = config['plot_params']['alpha']  # alpha value when plotting

    def plot_histogram(self, histogram_results, loader):
        """
        Plot histogram using results from "gen_histogram" method

        Inputs:
            histogram_results: output dictionary from "gen_histogram" function
        """
        # Processed data
        flux_raw = histogram_results['flux_raw']
        flux_bg_sub = histogram_results['flux_bg_sub']
        t_binedges = histogram_results['t_binedges']  # [s] temporal bin edges
        r_binedges = histogram_results['r_binedges']  # [m] range bin edges

        # Start plotting
        print('\nStarting to generate histogram plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi,
                         figsize=(self.figsize[0], self.figsize[1])
                         )
        ax = fig.add_subplot(111)
        if self.flux_bg_sub:
            mesh = ax.pcolormesh(t_binedges,
                                 r_binedges / 1e3,
                                 flux_bg_sub,
                                 cmap='viridis',
                                 norm=LogNorm(vmin=self.cbar_min,
                                              vmax=self.cbar_max)
                                 )
            fig.suptitle('CoBaLT Background-Subtracted Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
        else:
            mesh = ax.pcolormesh(t_binedges,
                                 r_binedges / 1e3,
                                 flux_raw,
                                 cmap='viridis',
                                 norm=LogNorm(vmin=flux_raw[flux_raw > 0].min(),
                                              vmax=flux_raw.max())
                                 )
            fig.suptitle('CoBaLT Backscatter Flux')
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Flux [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title('Scale {:.1f} m x {:.2f} s\n{} {}'.format(self.rbinsize, self.tbinsize,
                                                               "Low Gain" if loader.low_gain else "High Gain",
                                                               loader.timestamp))
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.plot_ylim else ax.set_ylim(
            [0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.plot_xlim else None
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        if self.save_img:
            img_fname = loader.generic_fname + '_hg' + '.png'
            loader.img_save_path = Path(loader.data_dir + loader.image_dir + loader.date) / img_fname
            fname = loader.get_unique_filename(loader.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
        plt.show()

    def plot_scatter(self, loader):
        """
        Display scatter plot of time tags
        """

        loader.load_chunk()

        ranges = np.concatenate([da.values.ravel() for da in loader.ranges_tot])
        shots_time = np.concatenate([da.values.ravel() for da in loader.shots_time_tot])

        # Start plotting
        print('\nStarting to generate scatter plot...')
        start = time.time()

        fig = plt.figure(dpi=self.dpi,
                         figsize=(self.figsize[0],
                                  self.figsize[1])
                         )
        ax = fig.add_subplot(111)
        ax.scatter(shots_time,
                   ranges / 1e3,
                   s=self.dot_size,
                   alpha=self.alpha,
                   linewidths=0
                   )
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.plot_ylim else ax.set_ylim([0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.plot_xlim else None
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        # ax.set_aspect('equal')
        ax.set_title(
            'CoBaLT Backscatter\n{} {}'.format("Low Gain" if loader.low_gain else "High Gain", loader.timestamp))
        plt.tight_layout()
        print('Finished generating plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        if self.save_img:
            print('Starting to save image...')
            start = time.time()
            img_fname = loader.generic_fname + '_scatter' + '.png'
            loader.img_save_path = Path(loader.data_dir + loader.image_dir + loader.date) / img_fname
            fname = loader.get_unique_filename(loader.img_save_path)
            fig.savefig(fname, dpi=self.save_dpi)
            print('Finished saving plot.\nTime elapsed: {:.1f} s'.format(time.time() - start))
        plt.show()

