from processing.processing_methods.data_loader import DataLoader
from processing.processing_methods.data_plotter import DataPlotter
from processing.processing_methods.deadtime_correct import DeadtimeCorrect

class DataPreprocessor:
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.plotter = DataPlotter(config)
        self.deadtime_correct = DeadtimeCorrect(config)

    def run(self):
        self.loader.preprocess()
        if self.deadtime_correct.apply_corrections:
            histogram_results = self.loader.gen_histogram()

            # Calculate Mueller correction
            mueller_results = self.deadtime_correct.mueller_correct(histogram_results, self.loader)

            # Calculate deadtime-model correction
            af_results = self.deadtime_correct.calc_af_hist(histogram_results, self.loader)
            dc_results = self.deadtime_correct.deadtime_model_correct(af_results, histogram_results)
            deadtime_bg_results = self.deadtime_correct.deadtime_bg_calc(self.loader, self.plotter)

            # Compare corrections
            self.deadtime_correct.plot_binwise_corrections(mueller_results, dc_results, deadtime_bg_results)
        else:
            if self.plotter.histogram:
                histogram_results = self.loader.gen_histogram()
                self.plotter.plot_histogram(histogram_results, self.loader)
            else:
                self.plotter.plot_scatter(self.loader)

