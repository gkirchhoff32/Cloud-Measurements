"""
Script to preprocess netCDF data from cloud measurements.
"""

from processing.processing_methods.data_loader_utils import DataLoader
from processing.processing_methods.data_plotter_utils import DataPlotter
from processing.processing_methods.deadtime_correct_utils import DeadtimeCorrect
from processing.processing_methods.data_processor_utils import DataProcessor
import re

# TODO: Take fft of raw and deadtime-corrected signals to quantify how much deadtime-periodic fluctuations are suppressed
# TODO: Look into less heterogeneous targets, such as smoke, rayleigh, or stratiform clouds


class Preprocessor:
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.plotter = DataPlotter(config)
        self.deadtime_correct = DeadtimeCorrect(config)
        self.processor = DataProcessor(config)

        self.config = config
        
    def run(self):
        self.loader.preprocess()
        if self.plotter.histogram:
            histogram_results = self.loader.gen_histogram()
            self.plotter.plot_histogram(histogram_results, self.loader)
        elif self.plotter.scatter:
            self.plotter.plot_scatter(self.loader)

    def switch_channel(self):
        """
        Switch loader from high- to low-gain channel
        """
        self.loader = DataLoader(self.config)
        self.loader.fname = re.sub(r'/Dev_(\d)_-', lambda m: f"/Dev_{1 - int(m.group(1))}_-", self.loader.fname)
        self.loader.preprocess()

