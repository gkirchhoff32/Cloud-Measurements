import os
from pathlib import Path
import re

from instrument.check_channel import is_low_gain
from instrument.timing import adjust_daylight_savings

def find_data_path(data_dir):
    """
    Automatically detect relevant file path for data loading based on OS.
    For Windows, checks common drive letters and user paths.
    For Linux, uses the data_dir as is or checks common mount points.
    """
    # Detect operating system
    if os.name == 'nt':  # Windows
        target_subdir = data_dir
        # Windows candidate paths
        candidate_roots = [
            Path("F:/"),
            Path("C:/Users/Grant"),
            Path("C:/Users/gkirc")
        ]

        for root in candidate_roots:
            candidate = root / target_subdir
            if candidate.exists():
                print(f"Detected Windows data directory: {candidate}")
                return str(candidate)

        raise FileNotFoundError("Could not locate the Windows data directory.")

    else:  # Linux/Unix
        # For Linux, first try the direct path
        linux_path = Path(data_dir)
        if linux_path.exists():
            print(f"Using Linux data directory: {linux_path}")
            return str(linux_path)

        # If direct path doesn't exist, check common Linux mount points
        linux_candidates = [
            Path("/home/grki4829/Data"),
            Path("/data"),
            Path("/mnt/data"),
        ]

        for candidate in linux_candidates:
            if candidate.exists():
                print(f"Detected Linux data directory: {candidate}")
                return str(candidate)

        raise FileNotFoundError("Could not locate the Linux data directory.")

def parse_filename(fname_nc):
    """
    Use standard naming convention from .ARSENL binary file to extract board number, time, and date.
    e.g., Dev_0_-_2025-09-13_21.47.46.ARSENL
    """

    match = re.match(r"Dev_(\d)_-_(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2}\.\d{2}).nc", fname_nc)
    if not match:
        raise ValueError(f"Filename format not recognized: {fname_nc}")

    # Pull out board number, date, and time.
    dev, date_str, time_str = match.groups()
    low_gain = is_low_gain(dev)

    # Convert to datetime if useful
    timestamp = adjust_daylight_savings(date_str, time_str)

    return low_gain, timestamp

def extract_chunk_number(path):
    """
    Sort files numerically by their chunk index
    """
    match = re.search(r'_(\d+)\.nc$', os.path.basename(path))
    return int(match.group(1)) if match else -1

def get_unique_filename(filename):
    """
    When saving data, if filename exists, then save to file name based on iterator

    Params:
        filename (str): Original save file name
    Returns:
        filename (str): New save file name
    """
    base, ext = os.path.splitext(filename)
    counter = 0
    filename = f"{base}_{counter}{ext}"
    while os.path.exists(filename):
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename


