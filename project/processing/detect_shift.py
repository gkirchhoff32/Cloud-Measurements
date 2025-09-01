


import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import yaml
from pathlib import Path
import xarray as xr
import os
import glob
from data_preprocessor import DataPreprocessor

config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

dp_hg = DataPreprocessor(config)
dp_lg = DataPreprocessor(config)
dp_hg.fname = r'/Dev_0_-_2025-08-05_02.14.22.ARSENL'
dp_lg.fname = r'/Dev_1_-_2025-08-05_02.14.22.ARSENL'
dp_lg.chunk_start = 0
dp_hg.preprocess()
dp_lg.preprocess()
histogram_results_hg = dp_hg.gen_histogram()
histogram_results_lg = dp_lg.gen_histogram()
dp_hg.plot_histogram(histogram_results_hg)
dp_lg.plot_histogram(histogram_results_lg)

print('Finished.')
