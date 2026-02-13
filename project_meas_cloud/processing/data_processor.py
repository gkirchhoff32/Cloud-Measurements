"""
Script to process data from sims or cloud measurements after preprocessing.
"""

from processing.processing_methods.deadtime_correct_utils import DeadtimeCorrect
from processing.processing_methods.data_processor_utils import DataProcessor
from processing.data_preprocessor_v2 import Preprocessor
from sims.gen_sim_data import GenerateData

class Processor:
    def __init__(self, config, use_sim, dpp):
        self.deadtime_correct = DeadtimeCorrect(config)
        self.processor = DataProcessor(config)

        self.config = config
        self.use_sim = use_sim
        self.loader = GenerateData(config) if self.use_sim else dpp.loader

        self.perform_corrections = config['process_params']['perform_corrections']

    def run(self, dpp):
        if self.perform_corrections:
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
                histogram_processing_results = self.processor.gen_histogram_processing(self.use_sim, self.loader, dpp)
                self.processor.generate_fits(self.use_sim, self.loader, dpp, histogram_processing_results)
                quit()
