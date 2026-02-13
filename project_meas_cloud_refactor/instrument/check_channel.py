"""
Objective: Check if measurement is from high- (Dev0) or low-gain (Dev1) channel
"""

def is_low_gain(device_number: str) -> bool:
    """
    Returns true if device number is 1 (or low gain). Returns false if otherwise (high gain).
    """
    return device_number == "1"