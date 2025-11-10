

class DataProcessor:
    def __init__(self, config):
        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

    def corrections_process(self, loader, plotter, deadtime_correct):
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
        fluxes_bg_sub = deadtime_correct.plot_binwise_corrections(mueller_results, dc_results, deadtime_bg_results)

        return fluxes_bg_sub
    
    def repeat_process(self, loader, plotter, deadtime_correct, num_seq):
        """
        Placeholder for data processing methods without corrections.
        """
        for i in range(num_seq):

            self.corrections_process(loader, plotter, deadtime_correct)
