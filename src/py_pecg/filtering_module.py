from collections.abc import Iterable  # pragma: no cover

import numpy as np  # pragma: no cover
import scipy.signal  # pragma: no cover

protocol_params = {  # pragma: no cover
    500: ["interpolate", "trim"],
    1000: ["interpolate", "trim"],
    200: ["interpolate", "trim", "downsample"],
    360: [
        "interpolate",
        "trim",
        "downsample",
        "interpolate",
        "trim",
        "downsample",
    ],
}

interpolation_params = {  # pragma: no cover
    500: [(2, 1, 0), (2, 3, 0), (2, 7, 0), (2, 7, 0), (2, 7, 0)],
    1000: [(4, 1, 0), (4, 3, 0), (4, 7, 0), (4, 7, 0), (4, 7, 0)],
    200: [
        False,
        (4, 3, 0.4),
        (4, 7, 0.4),
        (4, 7, 0.4),
        (4, 7, 0.4),
    ],
    360: [(6, 1, 0), (6, 3, 0), (6, 7, 0), (6, 7, 0), (6, 7, 0)],
}
trimming_params = {  # pragma: no cover
    500: 2,
    1000: 4,
    200: 4,
    360: 6,
}
downsample_params = {  # pragma: no cover
    500: False,
    1000: False,
    200: [[(0, 1), (3, 5), (4, 5), (1, 5), (1, 5)]],
    360: [
        [(3, 5), (0, 5), (4, 5), (2, 5), (2, 5)],
        [(1, 5), (4, 5), (3, 5), (3, 5), (3, 5)],
    ],
}

theoretical_max_scale = 5  # pragma: no cover


def interp(
    input_data: np.ndarray,
    interpolation_factor: int,
    lpf_length: int,
    cutoff_freq: float,
) -> np.ndarray:
    """
    Resample data at a higher rate using lowpass interpolation.
    Resamples the sequence in vector X at R times the original sample rate.
    The resulting resampled vector Y is R times longer, LENGTH(Y) = R*LENGTH(X).
    A symmetric filter allows the original data to pass through unchanged and
    interpolates between so that the mean square error between them and their
    ideal values is minimized.

    Args:
        input_data (np.ndarray): The target signal designated for interpolation.
        interpolation_factor (int): The interpolation factor of the resulting
            resampled vector.
        lpf_length (int): Half the number of original sample values used to
            perform the interpolation. For best results, use N no larger than 10.
        cutoff_freq (float): Cutoff Frequency. The signal is assumed to be band
            limited with cutoff_freq frequency 0 < CUTOFF <= 1.0.

    Raises:
        TypeError: Type validation for interpolation_factor, lpf_length and
            cutoff_freq.
        ValueError: Value validation for cutoff_freq.

    Returns:
        filter_coeff (np.ndarray): Returns the coefficients of the interpolation
            filter.
        output_data (np.ndarray): Returns the interpolated version of the target
            signal.

    Credits:
        Copyright 1988-2022 The MathWorks, Inc.
        "Programs for Digital Signal Processing", IEEE Press
        John Wiley & Sons, 1979, Chap. 8.1.
        This code is a modified version of the original.
    """
    # Argument Validation
    if cutoff_freq == 0:
        cutoff_freq = 0.5
    if isinstance(input_data, list):
        input_data = np.array(input_data)
    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if input_data.size == 0:
        raise ValueError("Input is an empty array.")
    if input_data.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a matrix.")

    if not isinstance(interpolation_factor, (int, np.integer)):
        raise TypeError("interpolation_factor must be a finite integer scalar.")
    if interpolation_factor < 1 or not np.isfinite(interpolation_factor):
        raise ValueError("interpolation_factor must be >= 1 and finite.")

    if not isinstance(lpf_length, (int, np.integer)):
        raise TypeError("lpf_length must be a finite integer scalar.")
    if lpf_length < 1 or not np.isfinite(lpf_length):
        raise ValueError("lpf_length must be >= 1 and finite.")

    if not isinstance(cutoff_freq, (int, float)):
        raise TypeError("cutoff_freq must be a finite scalar.")
    if cutoff_freq <= 0 or cutoff_freq > 1 or not np.isfinite(cutoff_freq):
        raise ValueError("cutoff_freq must be > 0, <= 1, and finite.")

    # Initial Setup
    input_data = input_data.flatten()
    initial_length = len(input_data)
    if not (2 * lpf_length + 1 < initial_length):
        raise ValueError("InvalidDimensions: 2*lpf_length + 1 must be < initial_length")

    resampled_length = int(interpolation_factor * initial_length)
    resampled_lpf_length = int(interpolation_factor * lpf_length)

    # Filter Design
    if cutoff_freq == 1:
        magnitude_vector = np.array([interpolation_factor, interpolation_factor, 0, 0])
        freq_vector = np.array(
            [
                0,
                1 / (2 * interpolation_factor),
                1 / (2 * interpolation_factor),
                0.5,
            ]
        )
    else:
        Nband = int(np.floor(0.5 * interpolation_factor))
        magnitude_vector = np.concatenate(
            ([interpolation_factor, interpolation_factor], np.zeros(2 * Nband))
        )
        a2r = cutoff_freq / (2 * interpolation_factor)
        freq_vector = np.zeros(2 * Nband + 2)
        freq_vector[1] = a2r
        for k in range(1, Nband + 1):
            freq_vector[2 * k] = k / interpolation_factor - a2r
            freq_vector[2 * k + 1] = k / interpolation_factor + a2r
        freq_vector[-1] = min(freq_vector[-1], 0.5)

    filter_coeff = scipy.signal.firls(
        2 * interpolation_factor * lpf_length + 1,
        2 * freq_vector,
        magnitude_vector,
    ).astype(np.float64)

    # Data Preparation
    resampled_data = np.zeros(resampled_length, dtype=np.float64)
    original_indices = np.arange(0, resampled_length, int(interpolation_factor))
    resampled_data[original_indices] = input_data

    # Handle Left Transients
    boundary_transients = np.zeros(2 * resampled_lpf_length)
    left_boundary_indices = np.arange(
        0, resampled_lpf_length * 2, int(interpolation_factor)
    )
    reversed_left_indices = np.arange(2 * int(lpf_length), 0, -1)
    boundary_transients[left_boundary_indices] = (
        2 * input_data[0] - input_data[reversed_left_indices]
    )
    initial_conditions = scipy.signal.lfilter(
        np.flip(filter_coeff), 1, np.flip(boundary_transients)
    )
    initial_conditions = np.flip(initial_conditions)

    # Filtering
    resampled_data, final_conditions = scipy.signal.lfilter(
        filter_coeff, 1, resampled_data, zi=initial_conditions
    )

    # Handle Main Signal
    main_signal_indices = np.arange(
        0, (initial_length - int(lpf_length)) * int(interpolation_factor)
    )
    main_signal_map_indices = np.arange(resampled_lpf_length, resampled_length)
    resampled_data[main_signal_indices] = resampled_data[main_signal_map_indices]

    # Handle Right Transients
    boundary_transients = np.zeros(2 * resampled_lpf_length)
    right_boundary_indices = np.arange(
        0, resampled_lpf_length * 2, int(interpolation_factor)
    )
    reversed_right_indices = np.arange(
        initial_length - 2, initial_length - 2 * int(lpf_length) - 2, -1
    )
    boundary_transients[right_boundary_indices] = (
        2 * input_data[initial_length - 1] - input_data[reversed_right_indices]
    )
    boundary_transients, _ = scipy.signal.lfilter(
        filter_coeff, 1, boundary_transients, zi=final_conditions
    )

    right_correction_indices = np.arange(
        resampled_length - resampled_lpf_length, resampled_length
    )
    right_map_indices = np.arange(0, resampled_lpf_length)
    resampled_data[right_correction_indices] = boundary_transients[right_map_indices]

    # Output Reshaping
    if input_data.ndim == 1 and input_data.shape[0] == input_data.size:
        output_data = np.reshape(resampled_data, (1, resampled_length))
    else:
        output_data = resampled_data

    return [output_data.flatten(), filter_coeff]


"""
Credits:
    Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
    Ana Paula Rocha, Pablo Laguna
    Original Publication: A wavelet-based ECG delineator:
    evaluation on standard databases
    Original DOI: 10.1109/TBME.2003.821031.
"""


def quadratic_splines_filterbank(fs: float):
    """
    Obtains the quadratic splines wavelet filter-bank filters from
    scale 1 to 4 as a function of the sampling frequency,
    in order to use filters with similar analog frequency behavior for
    different sampling frequencies.

    Args:
        fs (float): Sampling frequency of the target signal.

    Returns:
        quadratic_filter_list (list): A list of numpy arrays representing
        filters at different scales.

    Note:
        Initializing algorithme a trous (Decimation)
        Transfer Functions:
        Q1(w) = G(w);
        Q2(w) = H(w)G(2w);
        Q3(w) = H(w)H(2w)G(4w);
        Q4(w) = H(w)H(2w)H(4w)G(8w);
        Q5(w) = H(w)H(2w)H(4w)H(8w)G(16w);
    """
    if fs <= 0:
        raise ValueError("Sampling Frequency must be positive, and non-zero.")
    if fs not in (500, 1000, 360, 200):
        raise ValueError("Sampling Frequency must be either 200, 360, 500 or 1000.")

    def create_ordered_filter(base_filter, zeros_count):
        return np.kron(base_filter, np.array([1] + [0] * zeros_count))

    def apply_interpolation(quadratic_filter_list, interp_params):
        for i, (filt, params) in enumerate(zip(quadratic_filter_list, interp_params)):
            if params is False:  # Skip interpolation for this filter if params is False
                continue
            step, zero_count, fill_value = params
            quadratic_filter_list[i], _ = interp(
                np.concatenate(([0], filt, [0])), step, zero_count, fill_value
            )
        return quadratic_filter_list

    # Filter creation
    hpf_ord_1, lpf_ord_1 = 0.125 * np.array([1, 3, 3, 1]), 2 * np.array([1, -1])
    zeros_counts = [1, 3, 7, 15]
    hpf_filters = [hpf_ord_1] + [
        create_ordered_filter(hpf_ord_1, zeros) for zeros in zeros_counts
    ]
    lpf_filters = [lpf_ord_1] + [
        create_ordered_filter(lpf_ord_1, zeros) for zeros in zeros_counts
    ]
    quadratic_filter_list = [lpf_ord_1]
    current_filter = hpf_filters[0]
    for i in range(1, len(hpf_filters)):
        current_filter = (
            np.convolve(current_filter, hpf_filters[i - 1])
            if i != 1
            else current_filter
        )
        quadratic_filter_list.append(np.convolve(current_filter, lpf_filters[i]))

    # Filter Truncation
    filter_lengths = [max(np.argwhere(filt != 0)) for filt in quadratic_filter_list]
    for i, filt in enumerate(quadratic_filter_list):
        k = filter_lengths[i][0]
        quadratic_filter_list[i] = np.transpose(filt[: k + 1])

    # Filter Editing Protocol
    downsample_counter = 0
    for action in protocol_params[fs]:
        if action == "interpolate":
            apply_interpolation(quadratic_filter_list, interpolation_params[fs])
        elif action == "trim":
            for i in range(0, len(quadratic_filter_list)):
                quadratic_filter_list[i] = quadratic_filter_list[i][
                    1 : -trimming_params[fs]
                ]
        elif action == "downsample":
            for i in range(0, len(quadratic_filter_list)):
                temp_params = downsample_params[fs][downsample_counter][i]
                print(temp_params)
                quadratic_filter_list[i] = quadratic_filter_list[i][
                    temp_params[0] :: temp_params[1]
                ]
            downsample_counter += 1

    # Exceptional Case
    if fs == 200:
        quadratic_filter_list[0] = 1.1 * np.array([5 / 4, -5 / 4])

    return quadratic_filter_list


def wavelet_transform(
    signal: np.ndarray,
    quadratic_filter_list: list,
) -> np.ndarray:
    """
    Calculates the wavelet transform of a signal using quadratic
    spline wavelet. It calculates wavelets in scales of 1 to 4,
    Assuming the length of the filter-list is 5.
    Reference: Original DOI: 10.1109/TBME.2003.821031

    Args:
        signal (np.ndarray): The target signal designated for filtering.
        quadratic_filter_list (list): Quadratic spline filter bank.

    Returns:
        wavelet_transform_matrix (np.ndarray): Resultant Wavelet Transform
        Matrix.
    """
    # Validation
    if signal.size == 0:
        raise ValueError("Input is an empty array.")
    if not isinstance(quadratic_filter_list, Iterable):
        raise ValueError("quadratic_filter_list should be an iterable object.")
    if signal.squeeze().ndim > 1 or any(
        not isinstance(arr, np.ndarray) or arr.ndim != 1
        for arr in quadratic_filter_list
    ):
        raise ValueError("Input is a higher-dimensional array, not a matrix.")

    # Process
    wavelet_transform_matrix = np.zeros((len(signal), theoretical_max_scale))
    initial_param = 1
    for i, filt in enumerate(quadratic_filter_list):
        wavelet_transform_matrix[:, i] = np.transpose(
            scipy.signal.lfilter(filt, initial_param, signal)
        )
    return wavelet_transform_matrix


def create_filter(fs: float):
    """
    Creates quadratic spline filters using a variation on Mallat's Algorithm,
    algorithme à trous.

    Args:
        fs (float): Sampling frequency of the target signal.

    Returns:
        quadratic_filter_list (list): List of quadratic filters.
        quadratic_filter_lengths (list): List of lengths of the above quadratic
            filters.
        filter_decimation_values (list): List of decimated lengths of the above
            quadratic filters.
    """

    if fs <= 0:
        raise ValueError("Sampling Frequency must be positive, and non-zero.")
    if fs not in (500, 1000, 360, 200):
        raise ValueError("Sampling Frequency must be either 200, 360, 500 or 1000.")
    quadratic_filter_list = quadratic_splines_filterbank(fs)
    quadratic_filter_lengths = [len(q) for q in quadratic_filter_list]
    filter_decimation_values = [
        int(np.floor((k - 1) / 2)) for k in quadratic_filter_lengths
    ]
    return (
        quadratic_filter_list,
        quadratic_filter_lengths,
        filter_decimation_values,
    )


def filter_segment_of_signal(
    quadratic_filter_list: list,
    last_quadratic_filter_length: int,
    filter_decimation_values: list,
    current_samp: int,
    initial_samp: int,
    num_samples: int,
    segment_boundaries: list,
    signal_segment: np.ndarray,
):
    """
    Uses a Wavelet Transform on individual ECG signal excerpts, after
    generating quadratic-spline filter banks.

    Args:
        quadratic_filter_list (list): List of quadratic filters.
        quadratic_filter_length (int): Length of the largest (last) quadratic
            filter.
        filter_decimation_values (list): List of decimated lengths of the
            above quadratic filters.
        current_samp (int): Current sample analyzed within the ECG signal.
        initial_samp (int): Initial sample analyzed within the ECG signal.
        num_samples (int): Number of samples per excerpt for the filter bank
            corresponding to 2^16 samples at sf=250.
        segment_boundaries (list): First and last indices analyzed within the
            ECG signal.
        signal_segment (np.ndarray): The ECG signal designated for analysis.

    Returns:
        signal_segment (np.ndarray): The ECG signal designated for analysis.
        current_samp (int): Updated sample analyzed within the ECG signal.
        updated_samp (int): Updated current sample analyzed within the ECG
            signal.
        synchronized_wavelet_matrix (np.ndarray): Wavelet Transform Matrix,
            relevant to the current segment of the signal.
        threshold_matrix (np.ndarray): QRS detection threshold matrix.
        end_samp (int): Updated final sample analyzed within the ECG signal.
        initial_samp (int): Updated initial sample analyzed within the ECG
            signal.

    NOTE:
        First l5-1 samples are not correctly filtered (border effect).
        Last d5 samples are discarded in order to align all the filtered
        signals, taking into account the filter delays.
    """

    # Validation
    for component in (
        quadratic_filter_list,
        filter_decimation_values,
    ):
        if not isinstance(component, list):
            raise ValueError("The input must be a list.")
        if len(component) == 0:
            raise ValueError("The list cannot be empty.")
        for subcomponent in component:
            if not isinstance(subcomponent, (np.ndarray, int)):
                raise ValueError(f"{subcomponent} is not a numpy array.")

    for component in (
        current_samp,
        initial_samp,
        num_samples,
        last_quadratic_filter_length,
    ):
        if (not isinstance(component, int)) or (component < 0):
            raise ValueError(f"{component} must be a non-negative integer.")

    if signal_segment.size == 0:
        raise ValueError("Input is an empty array.")
    if signal_segment.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a matrix.")

    if not (
        isinstance(segment_boundaries, list)
        and all(isinstance(x, int) for x in segment_boundaries)
        and len(segment_boundaries) == 2
    ):
        raise ValueError(
            "segment_boundaries must be a non-empty list of ints or floats."
        )

    # Last sample excerpt analyzed
    updated_samp = current_samp
    segment_end = segment_boundaries[1]
    end_samp = min(initial_samp + num_samples - 1, segment_end)

    # Filter bank - Wavelet Matrix construction and parameters setting
    wavelet_transform_matrix = wavelet_transform(signal_segment, quadratic_filter_list)
    wavelet_transform_matrix = wavelet_transform_matrix[
        last_quadratic_filter_length - 1 :, 0:5
    ]

    # Synchronizing filtered signals at different scales
    d5 = filter_decimation_values[-1]
    synchronized_wavelet_matrix = np.zeros(
        (len(wavelet_transform_matrix) - d5, theoretical_max_scale)
    )
    for i, d in enumerate(filter_decimation_values):
        synchronized_wavelet_matrix[:, i] = wavelet_transform_matrix[
            d : d + len(wavelet_transform_matrix) - filter_decimation_values[-1],
            i,
        ]

    # Remove "incorrect" samples (see NOTE)
    current_samp = np.arange(
        initial_samp + last_quadratic_filter_length - 1, end_samp - d5 + 1
    )

    # Evaluate quality of the signal
    swt = np.power(synchronized_wavelet_matrix, 2)
    threshold_matrix = (
        0.5 * np.sqrt(np.median(swt, axis=0)) * np.ones((theoretical_max_scale, 1))
    )
    threshold_matrix[3, 0] = threshold_matrix[3, 0] * 2
    signal_segment = signal_segment[
        last_quadratic_filter_length - 1 : len(signal_segment) - d5
    ]

    # Specific segment of signal processed in this iteration
    threshold_matrix = threshold_matrix.T

    return (
        signal_segment,
        current_samp,
        updated_samp,
        synchronized_wavelet_matrix,
        threshold_matrix,
        end_samp,
        initial_samp,
    )
