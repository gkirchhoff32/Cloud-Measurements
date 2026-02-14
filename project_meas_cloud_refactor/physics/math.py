"""
Objective: Store commonly used math functions
"""

import numpy as np

def gaussian(x, A, mu, sigma, b):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b