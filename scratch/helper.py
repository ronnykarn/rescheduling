from scipy.signal import find_peaks


def find_peaks_in_window(data, window_length):
    """
    Find peaks within a sliding window of specified length.

    Parameters:
    - data: Array-like, the dataset in which to find peaks.
    - window_length: int, length of the sliding window.

    Returns:
    - peaks: List of indices where peaks were found.
    """

    # List to store peaks
    peaks = []

    # Slide the window across the data
    for start in range(0, len(data) - window_length + 1):
        end = start + window_length
        window_data = data[start:end]

        # Find peaks in the current window
        window_peaks = find_peaks(window_data)[0]

        # Convert window-relative indices to data-relative indices
        absolute_indices = [start + idx for idx in window_peaks]
        peaks.extend(absolute_indices)

    # Remove duplicates (peaks found in overlapping regions of windows)
    peaks = list(set(peaks))
    peaks.sort()

    return peaks
