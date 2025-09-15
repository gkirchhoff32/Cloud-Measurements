"""
Purpose: Measure timing delay between CoBaLT boards on startup.
Dataset: Brief measurement with laser firing and shut off part-way through.
General function: Load datasets from high- and low-gain channels. Return number of detected laser shots per channel and take difference.

09.15.2025
Grant Kirchhoff grant.kirchhoff@colorado.edu
"""


import yaml
from pathlib import Path

from data_preprocessor import DataPreprocessor

config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

dp_hg = DataPreprocessor(config)
dp_lg = DataPreprocessor(config)
dp_hg.fname = r'/Dev_0_-_2025-09-13_22.34.19.ARSENL'
dp_lg.fname = r'/Dev_1_-_2025-09-13_22.34.19.ARSENL'
dp_hg.preprocess()
dp_lg.preprocess()
last_sync_hg = dp_hg.last_sync_tot[-1]
last_sync_lg = dp_lg.last_sync_tot[-1]
print(last_sync_hg)
print(last_sync_lg)
print('Shot diff: {}'.format(last_sync_lg - last_sync_hg))
print('Time diff: {} s'.format((last_sync_lg - last_sync_hg) / dp_hg.PRF))

