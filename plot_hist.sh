#!/bin/bash

# make sure to install modules in virtual environment first
# 'python pip install -r ./cloudenv/requirements.txt'

cd /mnt/c/Users/Grant/Documents/ARSENL_Local/Cloud Measurements

source cloudenv/bin/activate

python ./library/plot_histogram.py

deactivate