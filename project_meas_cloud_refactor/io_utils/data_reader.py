"""
Objective: Functions used when loading .ARSENL files in chunks
"""

import pandas as pd
import glob
import os
import re

from utils.path_utils import extract_chunk_number

def stack_buffer(buffer, chunk):
    return pd.concat([buffer, chunk], ignore_index=True)

def trim_chunk(chunk):
    sync = chunk.loc[(chunk['overflow'] == 1) & (chunk['channel'] == 0)]
    if sync.empty:
        print(
            'Warning: Possible file chunk size too small. Did not find a laser shot event. Please use a '
            "larger chunk size if this wasn't the last chunk.")
        return 0
    elif len(sync) == 1:
        # If the sync length is only one, then reached the last laser shot. Finish.
        print('No more chunks to process.')
        return 1
    else:
        # Cut the chunk at the last 1,0 (sync event) row
        cut_idx = sync.index[-1]
        chunk_trim = chunk.iloc[:cut_idx]
        buffer = chunk.iloc[cut_idx:]

        return sync, chunk_trim, buffer

def remove_rollover(chunk_trim):
    # Clock rollover ("overflow", "channel" = 1,63). Max count is 2^25-1=33554431
    rollover = chunk_trim.loc[(chunk_trim['overflow'] == 1) & (chunk_trim['channel'] == 63)]

    chunk_fin = chunk_trim.drop(rollover.index)  # Remove rollover events
    chunk_fin = chunk_fin.reset_index(drop=True)  # Reset indices

    return chunk_fin

def locate_detect_sync(chunk_fin, chunk_iter):
    detect = chunk_fin.loc[
        (chunk_fin['overflow'] == 0) & (
                chunk_fin['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
    sync = chunk_fin.loc[
        (chunk_fin['overflow'] == 1) & (
                chunk_fin['channel'] == 0)]  # sync detection (laser pulse) ("overflow", "channel" = 1,0)

    # Ignore detections that precede first laser pulse event
    if chunk_iter == 0:
        start_idx = sync.index[0]
        detect = detect[detect.index > start_idx]

    return detect, sync

def get_chunk_files(preprocess_path, generic_fname):
    files = glob.glob(os.path.join(preprocess_path, f'{generic_fname}_*.nc'))
    files = sorted(files, key=extract_chunk_number)

    return files

def get_chunk_num(file_path):
    fname = os.path.basename(file_path)

    # Extract the final number before .nc
    match = re.search(r'_(\d+)\.nc$', fname)
    if match:
        chunk = int(match.group(1))
        return chunk
    else:
        return 0