"""
Purpose: Measure timing delay between CoBaLT boards on startup.
Dataset: Brief measurement with laser firing and shut off part-way through.
General function: Load datasets from high- and low-gain channels. Return number of detected laser shots per channel and take difference.

NOTE!!! For this particular script, make sure to select the "30-sec" datasets, such as "Dev_0_-_2025-09-13_22.34.19.ARSENL",
where the calibration segment is the only part of the measurement. In the actual preprocessing methods, the normal
measurement procedure is to start with a calibration segment then continue with the actual measurement. The script
automatically detects the calibration length and corrects for time delay. In a way, it means this script is outdated.

09.15.2025
Grant Kirchhoff grant.kirchhoff@colorado.edu
"""


import yaml
from pathlib import Path

from archived.data_preprocessor import DataPreprocessor

config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Load data. Make sure to use a calibration-test dataset where the laser was on before acquisition then turned off part-way through.
dp_hg = DataPreprocessor(config)
dp_lg = DataPreprocessor(config)
dp_hg.fname = r'/Dev_0_-_2025-09-13_22.34.19.ARSENL'
dp_lg.fname = r'/Dev_1_-_2025-09-13_22.34.19.ARSENL'
dp_hg.preprocess()
dp_lg.preprocess()

# Count total counts from each calibration-test dataset.
last_sync_hg = dp_hg.last_sync_tot[-1]
last_sync_lg = dp_lg.last_sync_tot[-1]
print(last_sync_hg)
print(last_sync_lg)
print('Shot diff: {}'.format(last_sync_lg - last_sync_hg))
print('Time diff: {} s'.format((last_sync_lg - last_sync_hg) / dp_hg.PRF))
