from pathlib import Path

from processing.processing_methods.data_loader import DataLoader
from processing.processing_methods.data_plotter import DataPlotter
from processing.processing_methods.deadtime_correct import DeadtimeCorrect

# TODO: Take fft of raw and deadtime-corrected signals to quantify how much deadtime-periodic fluctuations are suppressed


class DataPreprocessor:
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.plotter = DataPlotter(config)
        self.deadtime_correct = DeadtimeCorrect(config)

        self.config = config

    def run(self):
        self.loader.preprocess()
        if self.deadtime_correct.apply_corrections:
            histogram_results_hg = self.loader.gen_histogram()

            # Calculate Mueller correction
            mueller_results_hg = self.deadtime_correct.mueller_correct(histogram_results_hg, self.loader)

            # Calculate deadtime-model correction
            af_results_hg = self.deadtime_correct.calc_af_hist(histogram_results_hg, self.loader)
            dc_results_hg = self.deadtime_correct.deadtime_model_correct(af_results_hg, histogram_results_hg)
            deadtime_bg_results_hg = self.deadtime_correct.deadtime_bg_calc(self.loader, self.plotter)

            # Compare corrections
            fluxes_bg_sub_hg = self.deadtime_correct.plot_binwise_corrections(mueller_results_hg, dc_results_hg, deadtime_bg_results_hg)


            self.loader = DataLoader(self.config)
            self.loader.fname = r'/Dev_1_-_2025-09-13_22.36.08.ARSENL'
            self.loader.preprocess()
            histogram_results_lg = self.loader.gen_histogram()

            # Calculate Mueller correction
            mueller_results_lg = self.deadtime_correct.mueller_correct(histogram_results_lg, self.loader)

            # Calculate deadtime-model correction
            af_results_lg = self.deadtime_correct.calc_af_hist(histogram_results_lg, self.loader)
            dc_results_lg = self.deadtime_correct.deadtime_model_correct(af_results_lg, histogram_results_lg)
            deadtime_bg_results_lg = self.deadtime_correct.deadtime_bg_calc(self.loader, self.plotter)

            # Compare corrections
            fluxes_bg_sub_lg = self.deadtime_correct.plot_binwise_corrections(mueller_results_lg, dc_results_lg, deadtime_bg_results_lg)

            # Load both channels now
            self.deadtime_correct.plot_diff_overlap(fluxes_bg_sub_hg, fluxes_bg_sub_lg, self.loader)


            quit()
        else:
            if self.plotter.histogram:
                histogram_results = self.loader.gen_histogram()
                self.plotter.plot_histogram(histogram_results, self.loader)
            else:
                self.plotter.plot_scatter(self.loader)

