import numpy as np

"""
Credits:
    Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
    Ana Paula Rocha, Pablo Laguna
    Original Publication: A wavelet-based ECG delineator:
    evaluation on standard databases
    Original DOI: 10.1109/TBME.2003.821031.
"""


def find_local_modulus_minimum(
    x: np.ndarray,
) -> int:
    """
    Finds the first local minimum of the modulus of an array x,
    While truncating the edges (First and Final Indices).

    Args:
        x (np.ndarray): The target array.

    Returns:
        ind (int): The first index in x for which the first minimum is located.
    """
    if x.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a vector.")
    elif x.ndim == 0 or x.size == 0:
        raise ValueError("Input is empty.")

    x = np.abs(x)
    local_min = (x[1:-1] <= x[:-2]) & (x[1:-1] <= x[2:])
    temp_ind = np.argwhere(local_min)
    if temp_ind.size != 0:
        ind = min(temp_ind)
        ind = ind[0]
    else:
        ind = 0
    return ind


def search_onset_or_offset(
    search_index: int, signal: np.ndarray, set_threshold: float, flag: str
) -> int:
    """
    Finds the onset or the offset of a wave using the derivative method.

    Args:
        search_index (int): Position of the first relevant onset or offset in the wavelet.
        signal (np.ndarray): Wavelet Signal (in a single scale).
        set_threshold (float): Threshold Factor.
        flag (str): Valued "Onset" or "Offset" for each type of search.

    Returns:
        int: Onset index or Offset index, depending on the aforementioned flag.

    """
    if isinstance(signal, list):
        raise ValueError("Signal is a list, and not a np.ndarray type.")
    if not search_index.size or not signal.size:
        raise ValueError("One or more of the inputs are empty.")
    if set_threshold == 0:
        raise ZeroDivisionError("onset_threshold cannot be zero!")
    if signal.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a vector.")
    if not (flag == "Onset" or flag == "Offset"):
        raise ValueError(
            "Flag must match either an Onset search or an Offset search!"
        )

    else:
        if flag == "Onset":
            max_derivative = abs(signal[-1])
            processed_signal = np.flipud(signal[:-1])
            ind2 = find_local_modulus_minimum(np.flipud(signal[0:]))
            onset_factor = 1
            offset_factor = 0

        elif flag == "Offset":
            max_derivative = abs(signal[0])
            processed_signal = signal[1:]
            ind2 = find_local_modulus_minimum(signal[1:])
            onset_factor = 0
            offset_factor = 1

        ind1 = np.where(
            abs(processed_signal) < max_derivative / set_threshold
        )[0]

        if ind1.size != 0:
            ind1 = ind1[0]
        elif ind1.size == 0:
            ind1 = 0

        if not ind1 and not ind2:
            set_index = search_index + (offset_factor - onset_factor) * (
                len(signal) - 1
            )
        elif not ind1:
            set_index = (
                search_index
                - onset_factor * (ind2 + 1)
                + offset_factor * (ind2)
            )
        elif not ind2:
            set_index = (
                search_index
                - onset_factor * (ind1 + 1)
                + offset_factor * (ind1)
            )
        else:
            set_index = search_index + (offset_factor - onset_factor) * (
                min(ind1, ind2) + 1
            )
    return max(set_index, 0)


def search_onset(
    search_onset_index: int, signal: np.ndarray, onset_threshold: float
) -> int:
    """
    Finds the onset of a wave using the derivative method.
    This function is a wrapper function for search_onset_or_offset().

    Args:
        search_onset_index (int): Position of the first relevant onset in the wavelet.
        signal (np.ndarray): Wavelet signal (in a single scale).
        onset_threshold (float): Threshold Factor.

    Returns:
        onset_index (int): Resultant index of the wave onset.
    """
    onset_index = search_onset_or_offset(
        search_onset_index,
        signal,
        onset_threshold,
        "Onset",
    )
    return onset_index


def search_offset(
    search_offset_index: int, signal: np.ndarray, offset_threshold: float
) -> int:
    """
    Finds the offset of a wave using the derivative method.
    This function is a wrapper function for search_onset_or_offset().

    Args:
        search_offset_index (int): Position of the first relevant offset in the wavelet.
        signal (np.ndarray): Wavelet Signal (in a single scale).
        offset_threshold (float): Threshold Factor.

    Returns:
        offset_index (int): Resultant index of the wave offset.
    """
    offset_index = search_onset_or_offset(
        search_offset_index,
        signal,
        offset_threshold,
        "Offset",
    )
    return offset_index


def find_first_or_last_peak(signal: np.ndarray, time: int, flag: str) -> int:
    """
    Finds the first or the last peak of a signal, depending on the flag.

    Args:
        signal (np.ndarray): The signal array.
        time (int): Time relevant to the scale of the target signal.
        flag (str): Valued "First" or "Last" for each type of search.

    Returns:
        peak_index (int): Index of the first or the last peak of a signal,
                          depending on the aforementioned flag.
    """
    if signal.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a vector.")
    elif signal.size == 0:
        raise ValueError("Input an array which is not empty.")
    if time < 0:
        raise ValueError("Input a non-negative time.")
    if not (flag == "First" or flag == "Last"):
        raise ValueError("Flag must match either a First peak or a Last peak!")

    if flag == "Last":
        signal = np.flipud(signal)
        first_factor = 0
        last_factor = 1

    elif flag == "First":
        first_factor = 1
        last_factor = 0

    derivative = np.diff(signal)

    if flag == "Last":
        cero = min(np.argwhere((derivative[0:-1] * derivative[1:] <= 0)))

    elif flag == "First":
        cero = min(np.argwhere((derivative[0:-2] * derivative[1:-1] <= 0)))

    peak_index = time + (first_factor - last_factor) * (cero + 1)
    return peak_index


def find_last_peak(signal: np.ndarray, time: int) -> int:
    """
    Finds the last peak of a signal, nearest to the end.
    A wrapper function for find_first_or_last_peak().

    Args:
        signal (np.ndarray): The signal array.
        time (int): Time relevant to the scale of the target signal.

    Returns:
        last_peak_index (int): Index of the maximum in the target signal.
    """
    last_peak_index = find_first_or_last_peak(signal, time, "Last")
    return last_peak_index


def find_first_peak(signal: np.ndarray, time: int) -> int:
    """
    Finds the first peak of a signal, nearest to the beginning.
    A wrapper function for find_first_or_last_peak().

    Args:
        signal (np.ndarray): The signal array.
        time (int): Time relevant to the scale of the target signal.

    Returns:
        first_peak_index (int): Index of the maximum in the target signal.
    """
    first_peak_index = find_first_or_last_peak(signal, time, "First")
    return first_peak_index


def find_first_zero_crossing(x: np.ndarray):
    """
    Finds the index of the input vector in which the first
    zero crossing is located.

    Args:
        x (np.ndarray): The target array.

    Returns:
        int or np.ndarray:
            - int: The index where the first zero crossing occurs.
            - np.ndarray: An empty array if no zero crossing is found.
    """
    if x.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a vector.")
    elif x.size == 0:
        raise ValueError("Input an array which is not empty.")

    m = x[1:-1] * x[0:-2]
    if np.argwhere(m <= 0).size == 0:
        return np.array([])
    index = min(np.argwhere(m <= 0))
    if abs(x[index]) > abs(x[index + 1]):
        index = index + 1
    return index


def find_modulus_extrema(
    x: np.ndarray, first_samp: int, threshold: float, extrema_sign: float
) -> np.ndarray:
    """
    Finds the indices of local modulus maxima or minima in vector x
    where the modulus exceeds the specified threshold.

    Args:
        x (np.ndarray): The target signal array.
        first_samp (int): Analyze signal from first_samp sample.
        threshold (float): An amplitude threshold to consider maxima.
        extrema_sign (float): Sign of the maxima.
                       If the specified objective is 0, searches for
                       both positive and negative maxima.
                       If the specified objective is +1 or -1, searches
                       for modulus maxima positive or negative.

    Returns:
        int or np.ndarray:
            - indexes (np.ndarray): The indices of the modulus maxima.
            - np.ndarray: An empty array if no modulus maxima is found.
    """
    if x.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a vector.")
    elif x.size == 0:
        raise ValueError("Input an array which is not empty.")
    if first_samp < 0:
        raise ValueError("Input a non-negative first_samp.")

    lx = len(x) - 1

    if np.size(first_samp) < 2:
        first_sample = max(1, first_samp)
        last_sample = lx

    if lx > first_sample:
        s = np.sign(x)
        x = abs(x)

        sample_curr_idx = np.arange(first_sample, last_sample)
        sample_prev_idx = np.arange(first_sample - 1, last_sample - 1)
        sample_next_idx = np.arange(first_sample + 1, last_sample + 1)

        # FLOAT POINT PERCISION
        localmax = (
            (x[sample_curr_idx] >= x[sample_prev_idx])
            & (x[sample_curr_idx] > x[sample_next_idx])
            & (x[sample_curr_idx] >= threshold)
            & (s[sample_curr_idx] * extrema_sign >= 0)
        )  # if 0, it doesn't matter

        iAux = np.zeros_like(x, dtype=bool)
        iAux[sample_curr_idx] = localmax
        indexes = np.squeeze(np.argwhere(iAux))

    else:
        return np.array([])

    return indexes
