"""
Script to process data from cloud measurements after preprocessing.
"""

# from processing.processing_methods.data_loader_utils import DataLoader
# from processing.processing_methods.data_plotter_utils import DataPlotter
from processing.processing_methods.deadtime_correct_utils import DeadtimeCorrect
from processing.processing_methods.data_processor_utils import DataProcessor

class Processor:
    def __init__(self, config):
        self.deadtime_correct = DeadtimeCorrect(config)
        self.processor = DataProcessor(config)

        self.config = config

    def run(self, dpp):
        if dpp.deadtime_correct.apply_bin_corrections:
            if dpp.deadtime_correct.diff_overlap:
                fluxes_bg_sub_hg = dpp.processor.bin_corrections_process(dpp.loader, dpp.plotter, dpp.deadtime_correct)
                dpp.switch_channel()
                fluxes_bg_sub_lg = dpp.processor.bin_corrections_process(dpp.loader, dpp.plotter, dpp.deadtime_correct)

                # Load both channels
                overlap_results = dpp.deadtime_correct.plot_diff_overlap(fluxes_bg_sub_hg, fluxes_bg_sub_lg, dpp.loader)
                # r_binedges = overlap_results['r_binedges']
                # dr = r_binedges[1] - r_binedges[0]  # [m]
                # r_centers = r_binedges[:-1] + (dr / 2)  # [m]
                # self.deadtime_correct.parametric_fit(r_centers, overlap_results['d_olap_dc'])
            else:
                fluxes_bg_sub = dpp.processor.bin_corrections_process(dpp.loader, dpp.plotter, dpp.deadtime_correct)
                quit()
        else:

            histogram_results = dpp.loader.gen_histogram()
            self.processor.deadtime_fit(dpp.loader, dpp.deadtime_correct, histogram_results)
