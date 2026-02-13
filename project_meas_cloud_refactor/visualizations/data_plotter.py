"""
Objective: Script to handle plotting functions
"""

import time
import matplotlib.pyplot as plt
from pathlib import Path
import os

from utils.path_utils import get_unique_filename, find_data_path

class DataPlotter:
    def __init__(self, config):
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
        ax.set_ylim([self.ylim[0], self.ylim[1]]) if self.plot_ylim else ax.set_ylim([0, self.c / 2 / self.PRF / 1e3])
        ax.set_xlim([self.xlim[0], self.xlim[1]]) if self.plot_xlim else None
        ax.set_xlabel(timestamp.strftime("Time in seconds since %H:%M:%S %Z"))
        ax.set_ylabel('Range [km]')
        ax.set_title(
            timestamp.strftime(
                "CoBaLT Backscatter\n{} %Y-%m-%d %H:%M:%S %Z (UTC%z)".format("Low Gain" if low_gain else "High Gain")
            )
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