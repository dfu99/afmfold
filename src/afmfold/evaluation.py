import numpy as np

def compute_time_correlation(data, lag=1):
    # Convert 1D input to shape (N, 1)
    if data.ndim == 1:
        data = data[:, None]

    N, D = data.shape
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if lag >= N:
        raise ValueError("lag must be smaller than the number of time steps")

    # Replace infinities with NaN
    data = np.where(np.isfinite(data), data, np.nan)

    # Determine which time steps are valid (all dimensions are finite)
    valid_mask = np.all(np.isfinite(data), axis=1)

    # Extract consecutive valid blocks
    blocks = []
    start = None
    for i, valid in enumerate(valid_mask):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            blocks.append(slice(start, i))
            start = None
    if start is not None:
        blocks.append(slice(start, N))

    # Compute time correlation for each valid block
    num_sum = np.zeros(D)
    count_sum = np.zeros(D)

    for sl in blocks:
        block = data[sl]
        if len(block) <= lag:
            continue
        # Normalize each dimension (NaNs are already excluded)
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        normed = (block - mean) / std

        num_sum += np.nansum(normed[:-lag] * normed[lag:], axis=0)
        count_sum += np.sum(np.isfinite(normed[:-lag]) & np.isfinite(normed[lag:]), axis=0)

    # Final correlation value (normalized by valid data across blocks)
    with np.errstate(invalid='ignore', divide='ignore'):
        time_corr = num_sum / count_sum
    time_corr[~np.isfinite(time_corr)] = np.nan

    return time_corr
