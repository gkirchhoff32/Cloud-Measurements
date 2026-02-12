import pandas as pd

def calibrate_time(sync, chunk_trim, low_gain, fname, data_dir, date, chunksize, PRF):
    """
    Pure function version of your calibration logic.
    Returns:
        chunk_trim (DataFrame)
        shot_diff (float or None)
    """
    print('Starting to calculate calibration time shift...')

    # Calibration segment is determined by detecting large gap in sync pulse detections.
    sync_idx = sync.index
    gaps = sync_idx.to_series().diff().fillna(1)
    gap_start_idx = gaps.index[gaps.values.argmax() - 1]  # last laser shot before beginning of cal gap
    cal_sync = sync.loc[:gap_start_idx]  # cut everything past the beginning calibration section
    cal_shot_num = len(cal_sync)

    # If this is low-gain data, measure time delay using calibration section from high-gain
    # data too
    if low_gain:
        # Load small first chunk of high-gain file too
        hg_fname = fname.replace("Dev_1", "Dev_0")  # high-gain filename
        chunk_hg = next(pd.read_csv(data_dir + date + hg_fname,
                                    delimiter=',',
                                    chunksize=chunksize,
                                    dtype=int,
                                    on_bad_lines='skip',
                                    encoding_errors='ignore')
                        )
        sync_hg = chunk_hg.loc[(chunk_hg['overflow'] == 1) & (chunk_hg['channel'] == 0)]
        sync_hg_idx = sync_hg.index
        gaps_hg = sync_hg_idx.to_series().diff().fillna(1)
        gap_start_idx_hg = gaps_hg.index[gaps_hg.values.argmax() - 1]
        cal_sync_hg = sync_hg.loc[:gap_start_idx_hg]  # cut everything past the beginning calibration
        # section
        cal_shot_num_hg = len(cal_sync_hg)

        shot_diff = cal_shot_num - cal_shot_num_hg
        time_diff = shot_diff / PRF  # [s]
        print('Time diff between channels: {} s'.format(time_diff))
    else:
        shot_diff = None

    gap_end_idx = gaps.idxmax()  # first laser shot after end of calibration gap
    chunk_trim = chunk_trim.loc[gap_end_idx:]

    return chunk_trim, shot_diff