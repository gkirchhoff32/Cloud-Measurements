"""
Objective: store commonly used functions for physics-based conversions
"""

def range_to_time(ranges, c):
    """
    Convert ranges to equivalent time-of-flight.
    """
    return ranges / c * 2

def time_to_range(times, c):
    """
    Convert times-of-flight to equivalent ranges
    """
    return times * c / 2

def convert_flux(cnts, binsize, N):
    """
    Calculate flux from counts.
    Args:
        cnts (array-like or float): Number of detected counts in each bin.
        binsize (float): Duration of each range bin in seconds.
        N (int or float): Number of shots (pulses) contributing to each bin.

    Returns:
        array-like or float: Flux in Hz (counts per second), computed as
        cnts / (N * binsize)
    """
    return cnts / (N * binsize)
