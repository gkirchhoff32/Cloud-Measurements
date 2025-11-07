from pathlib import Path

from processing.processing_methods.data_loader import DataLoader
from processing.processing_methods.data_plotter import DataPlotter
from processing.processing_methods.deadtime_correct import DeadtimeCorrect
from processing.processing_methods.data_processor import DataProcessor
import re

# TODO: Take fft of raw and deadtime-corrected signals to quantify how much deadtime-periodic fluctuations are suppressed
# TODO: Look into less heterogeneous targets, such as smoke, rayleigh, or stratiform clouds


class DataPreprocessor:
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.plotter = DataPlotter(config)
        self.deadtime_correct = DeadtimeCorrect(config)
        self.processor = DataProcessor(config)

        self.config = config

    def switch_channel(self):
        """
        Switch loader from high- to low-gain channel
        """
        self.loader = DataLoader(self.config)
        self.loader.fname = re.sub(r'/Dev_(\d)_-', lambda m: f"/Dev_{1 - int(m.group(1))}_-", self.loader.fname)
        self.loader.preprocess()
        
    def run(self):
        self.loader.preprocess()
        if self.deadtime_correct.apply_corrections:
            fluxes_bg_sub_hg = self.processor.corrections_process(self.loader, self.plotter, self.deadtime_correct)
            self.switch_channel()
            fluxes_bg_sub_lg = self.processor.corrections_process(self.loader, self.plotter, self.deadtime_correct)

            # Load both channels now
            overlap_results = self.deadtime_correct.plot_diff_overlap(fluxes_bg_sub_hg, fluxes_bg_sub_lg, self.loader)
            # r_binedges = overlap_results['r_binedges']
            # dr = r_binedges[1] - r_binedges[0]  # [m]
            # r_centers = r_binedges[:-1] + (dr / 2)  # [m]
            # self.deadtime_correct.parametric_fit(r_centers, overlap_results['d_olap_dc'])
        else:
            if self.plotter.histogram:
                histogram_results = self.loader.gen_histogram()
                self.plotter.plot_histogram(histogram_results, self.loader)
            else:
                self.plotter.plot_scatter(self.loader)

