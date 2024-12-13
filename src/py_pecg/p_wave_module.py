from collections.abc import Iterable

import numpy as np
from py_pecg.utility_module import (
    find_first_zero_crossing,
    find_modulus_extrema,
    search_offset,
    search_onset,
)


def find_secondary_extrema(
    start_idx: int,
    end_idx: int,
    wavelet_matrix: np.ndarray,
    scale: int,
    direction: float,
    relevance_threshold: float,
    primary_amplitude: float,
):
    """
    Identifies and filters secondary extrema within a specified range
    of a wavelet matrix based on their relevance relative to a primary
    amplitude and threshold.

    Args:
        start_idx (int): Start index of the analyzed window in the signal.
        end_idx (int): Final index of the analyzed window in the signal.
        wavelet_matrix (np.ndarray): Wavelet Transform Matrix.
        scale (int): Current scale of wavelet being used.
        direction (float): Flag for maximum (+1) or minimum (-1).
        relevance_threshold (float): Threshold for detecting extrema.
        primary_amplitude (float): Amplitude being examined.

    Returns:
        extrema_pos: Extrema positions (int or list)
        extrema_val: Extrema values (float or list)

    """
    # Validation
    if start_idx >= end_idx or not (start_idx >= 0 and end_idx > 0):
        raise ValueError("The integer indices have to apply: start < end")
    if scale < 0:
        raise ValueError("scale must be a non-negative integer.")
    if np.squeeze(wavelet_matrix).size <= 1:
        raise ValueError("Wavelet matrix must be a multi-dimensional array")
    if direction not in (1, -1):
        raise ValueError("Direction flag must equal to +1 or -1.")
    if relevance_threshold <= 0:
        raise ValueError("relevance_threshold must be a positive float.")
    if primary_amplitude <= 0:
        raise ValueError("primary_amplitude must be a positive float.")

    # Finding extrema positions
    sub_matrix = wavelet_matrix[start_idx:end_idx, scale]
    extrema_positions = start_idx + find_modulus_extrema(sub_matrix, 2, 0, direction)

    # If extrema was found, categorize it
    if extrema_positions.size > 0:
        extrema_values = wavelet_matrix[extrema_positions, scale]
        if np.isscalar(extrema_values):
            extrema_val = extrema_values
            extrema_pos = extrema_positions
        else:
            extrema_func = max if direction == 1 else min
            extrema_val = extrema_func(extrema_values)
            extrema_index = (
                np.argmax(extrema_values)
                if direction == 1
                else np.argmin(extrema_values)
            )
            extrema_pos = extrema_positions[extrema_index]
    else:
        extrema_val = 0
        extrema_pos = []

    # Filter out irrelevant extrema
    if abs(extrema_val) < relevance_threshold * primary_amplitude:
        return [], 0

    return extrema_pos, extrema_val


# Note: this function appears only once, but this distinction helps
# Validating the process through more rigorous testing.
def refine_simple_p_wave(
    onset: int,
    offset: int,
    wavelet_matrix: np.ndarray,
    alternative_scale: int,
    current_scale: int,
    changes: list,
):
    """
    Attempts to find a simple p-wave by characterizing a zero-crossing
    point in the signal.

    Args:
        onset (int): Start index of the signal segment.
        offset (int): End index of the signal segment.
        wavelet_matrix (np.ndarray): Wavelet Transform Matrix.
        alternative_scale (int): Alternative scale of wavelet being used.
        current_scale (int): Current scale of wavelet being used.
        changes (list): List of flags controlling the refinement behavior.

    Returns:
        (int or None, depending on the analysis result)
        p_wave_peaks: The detected p-wave peaks position or None if not found.
        estimated_onset: Estimated onset of the p-wave.
        estimated_offset: Estimated offset of the p-wave.
    """
    # Validation
    if onset >= offset or not (onset >= 0 and offset > 0):
        raise ValueError("The integer indices have to apply: onset < offset")
    if alternative_scale < 0 or current_scale < 0:
        raise ValueError("scales must be non-negative integers.")
    if np.squeeze(wavelet_matrix).size <= 1:
        raise ValueError("Wavelet matrix must be a multi-dimensional array")

    ind = find_first_zero_crossing(wavelet_matrix[onset:offset, alternative_scale])

    if ind.size == 0 and "use_current_scale" in changes:
        ind = find_first_zero_crossing(wavelet_matrix[onset:offset, current_scale])
    if ind.size > 0:
        ind = np.squeeze(ind)
        p_wave_peaks = onset + ind - 1
        estimated_onset = onset
        estimated_offset = offset
    else:
        # Unknown case: unable to classify P wave
        p_wave_peaks, estimated_onset, estimated_offset = None, None, None

    return p_wave_peaks, estimated_onset, estimated_offset


def refine_complex_p_wave(
    extrema_positions: list,
    wavelet_matrix: np.ndarray,
    alternative_scale: int,
    current_scale: int,
    changes: list,
):
    """
    Sharpens p-wave onset and offset positions by analyzing zero-
    crossing of the wavelet matrix, synchronized with the ECG signal.

    Args:
        extrema_positions (list): A list of pre-located extrema to analyze.
        wavelet_matrix (np.ndarray): Wavelet Transform Matrix.
        alternative_scale (int): Alternative scale of wavelet being used.
        current_scale (int): Current scale of wavelet being used.
        changes (list): List of flags controlling the refinement behavior.

    Returns:
        (int or None, depending on the analysis result)
        p_wave_peaks: The detected p-wave peaks position or None if not found.
        estimated_onset: Estimated onset of the p-wave.
        estimated_offset: Estimated offset of the p-wave.
        secondary_peak: Auxilary P-wave peak.
    """
    # Validation
    if alternative_scale < 0 or current_scale < 0:
        raise ValueError("scales must be non-negative integers.")
    if np.squeeze(wavelet_matrix).size <= 1:
        raise ValueError("Wavelet matrix must be a multi-dimensional array")
    if not isinstance(extrema_positions, Iterable):
        raise ValueError("extrema_positions should be an iterable object.")

    extrema_positions = sorted(extrema_positions)
    p_wave_peaks, estimated_onset, estimated_offset, secondary_peak = (
        None,
        None,
        None,
        None,
    )

    def refine_position(start, end, scale):
        """
        Refines the analysis by using an alternative scale
        for the zero-crossing analysis.
        """
        ind = find_first_zero_crossing(wavelet_matrix[start:end, scale])
        if ind.size == 0 and "use_current_scale" in changes:
            ind = find_first_zero_crossing(wavelet_matrix[start:end, current_scale])
        return ind

    # Refine onset
    if extrema_positions[1] - extrema_positions[0] > 2:
        ind = refine_position(
            extrema_positions[0], extrema_positions[1], alternative_scale
        )
        if ind.size > 0:
            ind = np.squeeze(ind)
            p_wave_peaks = extrema_positions[0] + ind - 1
            estimated_onset = extrema_positions[0]

    # Refine offset
    if extrema_positions[2] - extrema_positions[1] > 2:
        ind = refine_position(
            extrema_positions[1], extrema_positions[2], alternative_scale
        )
        if ind.size > 0:
            ind = np.squeeze(ind)
            secondary_peak = extrema_positions[1] + ind - 1
            estimated_offset = extrema_positions[2]

    return p_wave_peaks, estimated_onset, estimated_offset, secondary_peak


def detect_p_wave_features(
    has_p_wave: bool,
    wavelet_matrix: np.ndarray,
    window_start: int,
    window_end: int,
    current_scale: int,
    qrs_times: np.ndarray,
    beat_indices: np.ndarray,
    i: int,
    freq: int,
    amplitude_threshold: float,
):
    """
    Detects the existance of a p-wave within an ECG signal segment.

    Args:
        has_p_wave (bool): A descriptor of whether the segment contains
        a p-wave.
        wavelet_matrix (np.ndarray): Wavelet Transform Matrix.
        window_start (int): Starting index of the analyzed window.
        window_end (int): End index of the analyzed window.
        current_scale (int): Current scale of wavelet being used.
        qrs_times (np.ndarray): Pre-detected qrs durations of the segment.
        beat_indices (np.ndarray): Indices of heartbeats in the segment.
        i (int): Index of current heartbeat analyzed.
        freq (int): Sampling frequency of the ECG signal.
        amplitude_threshold (float): Threshold for the P-wave peak detection.

    Returns:
        TYPE: DESCRIPTION.
    """

    def find_primary_extrema(
        wavelet_matrix, window_start, window_end, current_scale, direction
    ):
        """
        Pinpoints the most significant local peak within a defined window
        and scale of a wavelet-transformed signal, handling cases where
        no clear local extrema are present.
        """
        start_value = wavelet_matrix[window_start, current_scale]
        end_value = wavelet_matrix[window_end, current_scale]
        sub_matrix = wavelet_matrix[window_start + 1 : window_end, current_scale]
        positions = window_start + find_modulus_extrema(sub_matrix, 2, 0, direction) + 1

        length_positions = (
            len(positions) if isinstance(positions, (list, np.ndarray)) else 1
        )

        if length_positions > 1:
            extrema_func = np.argmax if direction > 0 else np.argmin
            positions = positions[
                extrema_func(wavelet_matrix[positions, current_scale])
            ]
        else:
            positions = (
                next(iter(positions), positions)
                if isinstance(positions, (list, np.ndarray)) and len(positions) > 0
                else positions
            )

        if length_positions == 0:
            if direction > 0:  # Maxima
                if start_value >= end_value and start_value > 0:
                    return window_start, start_value
                elif end_value > 0:
                    return window_end, end_value
            else:  # Minima
                if start_value <= end_value and start_value < 0:
                    return window_start, start_value
                elif end_value < 0:
                    return window_end, end_value
            return None, 0

        amplitude = wavelet_matrix[positions, current_scale]
        return positions, amplitude

    def compute_segment_rms(segment, current_scale):
        """
        Calculates the energy within a segment of the signal,
        at the specified scale, within the time domain.
        """
        return np.sqrt(np.mean(segment[:, current_scale]) ** 2)

    def compute_rms(wavelet_matrix, qrs_times, current_scale, beat_indices, i):
        """
        Identifies a specific heartbeat or interval in the wavelet matrix,
        and calculates the energy within that segment of the signal,
        at the specified scale.
        """
        if (
            len(beat_indices) != 2
            or beat_indices[0] != beat_indices[1]
            or (len(qrs_times) == 1 and beat_indices[i] != 1)
        ):
            start_idx = qrs_times[i - 1] if i != 0 else 0
            end_idx = qrs_times[i] + 1
        else:
            start_idx = qrs_times[0] if beat_indices[i] != 0 else 0
            end_idx = qrs_times[1] + 1 if beat_indices[i] != 0 else qrs_times[0] + 1

        segment = wavelet_matrix[start_idx:end_idx]
        return compute_segment_rms(segment, current_scale)

    # Validation
    if np.squeeze(wavelet_matrix).size <= 1:
        raise ValueError("Wavelet matrix must be a multi-dimensional array")
    if current_scale < 0:
        raise ValueError("scale must be a non-negative integer.")
    if i < 0:
        raise ValueError("index i must be a non-negative integer.")
    if not all(isinstance(arg, Iterable) for arg in (qrs_times, beat_indices)):
        raise ValueError("extrema_positions should be an iterable object.")
    if freq <= 0:
        raise ValueError("scale must be a positive integer.")
    if amplitude_threshold <= 0:
        raise ValueError("Amplitude threshold must be a non-negative float.")

    # Detection
    max_position, max_amplitude = find_primary_extrema(
        wavelet_matrix, window_start, window_end, current_scale, +1
    )
    min_position, min_amplitude = find_primary_extrema(
        wavelet_matrix, window_start, window_end, current_scale, -1
    )
    abs_max_amplitude = abs(max_amplitude)
    abs_min_amplitude = abs(min_amplitude)

    # Calculate signal RMS
    signal_rms = compute_rms(wavelet_matrix, qrs_times, current_scale, beat_indices, i)

    # Check if a P wave exists
    has_p_wave = (
        max_position is not None
        and min_position is not None
        and abs(max_position - min_position) < 0.11 * freq
        and abs_max_amplitude > amplitude_threshold * signal_rms
        and abs_min_amplitude > amplitude_threshold * signal_rms
    )
    has_p_wave = bool(has_p_wave)
    return (
        has_p_wave,
        abs_max_amplitude,
        abs_min_amplitude,
        max_position,
        min_position,
    )


def find_p_wave(
    signal_info: dict,
    sample_excerpt: np.ndarray,
    qrs_times: np.ndarray,
    locations: dict,
    wavelet_matrix: np.ndarray,
    beat_indices: list,
    last_annotation,
    fs: float,
    config_params: dict,
):
    """
    Identifies P wave in an ECG signal excerpt, as well as its onset
    and offset.

    Args:
        signal_info (dict): Internal Dictionary containg information
        regarding the overall ECG signal.
        sample_excerpt (np.ndarray): Samples included in the current excerpt
        (borders excluded).
        qrs_times (np.ndarray): QRS times in inedexes refering the interval
        included in the current excerpt (borders excluded).
        locations (dict): Dictionary containing information relevant to the
        P, QRS and T intervals in the ECG signal excerpt.
        wavelet_matrix (np.ndarray): Matrix with WT scales 1 to 5.
        beat_indices (list): First and last beat detected in the original
        ECG signal.
        last_annotation (int): Last annotation given to the sample.
        fs (float): Sampling frequency of the ECG signal.
        config_params (dict): Configuration dictionary for the algorithm.

    Returns:
        locations (dict): Dictionary containing information relevant to the
        P, QRS and T intervals in the ECG signal excerpt.
        estimated_onset_array (np.ndarray): First relevant slope associated to
        all P complexes (WT extrema).
        estimated_offset_array (np.ndarray): Last relevant slope associated to
        all P complexes (WT extrema).

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """

    def to_nan(x):
        return np.nan if not x else x

    # Validation Section
    # signal_info
    if not isinstance(signal_info, dict):
        raise ValueError("signal_info must be a dictionary.")

    # locations
    required_locations_keys = [
        "QRS_Complex_Onset",
        "P_Wave_Onset",
        "P_Wave_Offset",
        "P_Wave_Peak",
        "P_Wave_Type",
    ]
    if not isinstance(locations, dict):
        raise ValueError("locations must be a dictionary.")
    if not all(key in locations for key in required_locations_keys):
        raise ValueError(f"locations must contain the keys: {required_locations_keys}")

    # config_params
    required_config_params_keys = [
        "P Module",
        "amplitude_threshold_P",
        "relevance_threshold_P",
        "onset_percision_p",
        "offset_percision_p",
        "final_window_tol_P",
        "initial_window_tol_P",
        "min_window_length_P",
        "P_bound_tol",
    ]
    if not isinstance(config_params, dict):
        raise ValueError("config_params must be a dictionary.")
    if "P Module" not in config_params:
        raise ValueError("config_params must contain the key 'P Module'.")
    if not all(
        key in config_params["P Module"] for key in required_config_params_keys[1:]
    ):
        raise ValueError(
            f"config_params['P Module'] must contain:{required_config_params_keys[1:]}"
        )

    # sample_excerpt
    if not isinstance(sample_excerpt, np.ndarray):
        raise ValueError("sample_excerpt must be a NumPy array.")
    if isinstance(sample_excerpt, np.ndarray):
        if sample_excerpt.squeeze().ndim > 1:
            raise ValueError("sample_excerpt must be a 1D array.")
    if isinstance(sample_excerpt, np.ndarray):
        if not np.issubdtype(sample_excerpt.dtype, np.number):
            raise ValueError("sample_excerpt array must contain numerical values.")

    # qrs_times
    if not isinstance(qrs_times, np.ndarray):
        raise ValueError("qrs_times must be a NumPy array.")
    if isinstance(qrs_times, np.ndarray):
        if qrs_times.squeeze().ndim > 1:
            raise ValueError("qrs_times must be a 1D array.")
    if isinstance(qrs_times, np.ndarray):
        if not np.issubdtype(qrs_times.dtype, np.number):
            raise ValueError("qrs_times array must contain numerical values.")

    # wavelet_matrix
    if not isinstance(wavelet_matrix, np.ndarray):
        raise ValueError("qrs_times must be a NumPy array.")

    # beat_indices
    if not isinstance(beat_indices, list) or not all(
        isinstance(i, int) for i in beat_indices
    ):
        raise ValueError("beat_indices must be a list of integers.")

    # last_annotation
    if not isinstance(last_annotation, int):
        raise ValueError("last_annotation must be an integer.")

    # Array initialization
    p_wave_peaks, secondary_peak = [], []
    estimated_onset, estimated_offset = [], []
    refined_onset, refined_offset = [], []
    estimated_onset_array, estimated_offset_array = [], []
    changes = ["use_current_scale", "refine_onset_offset", "reset_refinement"]
    if not signal_info:
        return None, None, None

    # Parameter assignment
    freq = fs
    amplitude_threshold = config_params["P Module"]["amplitude_threshold_P"]
    relevance_threshold = config_params["P Module"]["relevance_threshold_P"]
    onset_percision = config_params["P Module"]["onset_percision_p"]
    offset_percision = config_params["P Module"]["offset_percision_p"]
    has_p_wave = 0
    abs_max_amplitude = abs_min_amplitude = None
    max_position = min_position = None

    if beat_indices:
        beat_start, beat_end = beat_indices[0], beat_indices[-1]
        unique_beats = np.unique(np.arange(beat_start, beat_end))

        for i in range(len(unique_beats)):
            (
                max_pre_peak_positions,
                min_pre_peak_position,
                min_post_peak_positions,
                max_post_peak_positions,
            ) = ([], [], [], [])

            # Prevent p-wave overlapping with QRS waves/t waves
            qrs_offset = (
                locations["QRS_Complex_Onset"][i + beat_indices[0]]
                - sample_excerpt[0]
                + 1
            )
            qrs_valid = (
                not np.isnan(locations["QRS_Complex_Onset"][i + beat_indices[0]])
                and qrs_offset < qrs_times[i]
            )
            if qrs_valid:
                qrson = qrs_offset
            else:
                final_window_tol = config_params["P Module"]["final_window_tol_P"]
                qrson = qrs_times[i] - round(final_window_tol * freq)

            # Configuring initial window
            initial_tol = config_params["P Module"]["initial_window_tol_P"]
            initial_window = round(initial_tol * freq) - 1
            prev_index_offset = i - 1 + beat_indices[0] - 1
            has_prev_index = prev_index_offset > 0
            if has_prev_index:
                previous_index = prev_index_offset
                sample_offset = sample_excerpt[0] - 1
                adjustment_keys = [
                    "T_Wave_Offset",
                    "T_Wave_Peak",
                    "QRS_Complex_Offset",
                    "qrs",
                ]
                adjusted_windows = []
                for key in adjustment_keys:
                    location = np.squeeze(locations[key])[previous_index]
                    if not np.isnan(location):
                        offset = location - sample_offset
                        adjusted_window = qrson - offset
                    else:
                        adjusted_window = initial_window
                    adjusted_windows.append(adjusted_window)
                initial_window = min([initial_window] + adjusted_windows)

            # Configuring final window
            final_tol = config_params["P Module"]["final_window_tol_P"]
            final_window = round(final_tol * freq) - 1
            last_annotation = last_annotation - sample_excerpt[0] + 2
            window_start = round(
                max(
                    1,
                    qrson - initial_window + 1,
                    (last_annotation + 1 if i == 1 and last_annotation >= 1 else 0),
                )
            )
            window_start = round(window_start)
            window_end = round(max(1, qrson - final_window + 1)) - 1

            min_window_len = config_params["P Module"]["min_window_length_P"]
            if window_end - window_start >= min_window_len * freq:
                current_scale = 3
                (
                    has_p_wave,
                    abs_max_amplitude,
                    abs_min_amplitude,
                    max_position,
                    min_position,
                ) = detect_p_wave_features(
                    has_p_wave,
                    wavelet_matrix,
                    window_start,
                    window_end,
                    current_scale,
                    qrs_times,
                    beat_indices,
                    i,
                    freq,
                    amplitude_threshold,
                )
                if not has_p_wave:
                    current_scale = 4
                    (
                        has_p_wave,
                        abs_max_amplitude,
                        abs_min_amplitude,
                        max_position,
                        min_position,
                    ) = detect_p_wave_features(
                        has_p_wave,
                        wavelet_matrix,
                        window_start,
                        window_end,
                        current_scale,
                        qrs_times,
                        beat_indices,
                        i,
                        freq,
                        amplitude_threshold,
                    )
            else:
                has_p_wave = 0

            # Detecting secondary extrema
            has_valid_changes = (
                "apply_secondary_detection" in changes or current_scale == 3
            )
            alternative_scale = 2 if "final_validation" in changes else current_scale

            if has_p_wave and has_valid_changes and "refine_onset_offset" in changes:
                is_max_first = abs_max_amplitude > abs_min_amplitude
                positions_valid = (
                    max_position < min_position
                    if is_max_first
                    else min_position < max_position
                )

                minaim = minpim = maxaim = maxpim = None
                min_pre_peak_position = min_post_peak_positions = (
                    max_pre_peak_positions
                ) = max_post_peak_positions = []
                if positions_valid:
                    if is_max_first:
                        min_pre_peak_position, minaim = find_secondary_extrema(
                            window_start + 1,
                            max_position + 1,
                            -1,
                            abs_max_amplitude,
                            wavelet_matrix,
                            current_scale,
                            relevance_threshold,
                        )
                        min_post_peak_positions, minpim = find_secondary_extrema(
                            max_position + 1,
                            window_end + 1,
                            -1,
                            abs_max_amplitude,
                            wavelet_matrix,
                            current_scale,
                            relevance_threshold,
                        )
                    else:
                        max_pre_peak_positions, maxaim = find_secondary_extrema(
                            window_start + 1,
                            min_position + 1,
                            1,
                            abs_min_amplitude,
                            wavelet_matrix,
                            current_scale,
                            relevance_threshold,
                        )
                        max_post_peak_positions, maxpim = find_secondary_extrema(
                            min_position + 1,
                            window_end + 1,
                            1,
                            abs_min_amplitude,
                            wavelet_matrix,
                            current_scale,
                            relevance_threshold,
                        )

                # Refinement/Classification of P-waves
                if (
                    not max_pre_peak_positions
                    and not min_pre_peak_position
                    and not max_post_peak_positions
                    and not min_post_peak_positions
                ):
                    p_type, onset, offset = (
                        (1, max_position, min_position)
                        if max_position < min_position
                        else (0, min_position, max_position)
                    )
                    if abs(offset - onset) > 2:
                        p_wave_peaks, estimated_onset, estimated_offset = (
                            refine_simple_p_wave(
                                onset,
                                offset,
                                wavelet_matrix,
                                alternative_scale,
                                current_scale,
                                changes,
                            )
                        )

                elif max_pre_peak_positions or max_post_peak_positions:
                    p_type = 4
                    (
                        p_wave_peaks,
                        estimated_onset,
                        estimated_offset,
                        secondary_peak,
                    ) = refine_complex_p_wave(
                        [
                            min_position,
                            max_position,
                            max_pre_peak_positions,
                            max_post_peak_positions,
                        ],
                        wavelet_matrix,
                        alternative_scale,
                        current_scale,
                        changes,
                    )
                elif min_pre_peak_position or min_post_peak_positions:
                    p_type = 5
                    (
                        p_wave_peaks,
                        estimated_onset,
                        estimated_offset,
                        secondary_peak,
                    ) = refine_complex_p_wave(
                        [
                            min_position,
                            max_position,
                            min_pre_peak_position,
                            min_post_peak_positions,
                        ],
                        wavelet_matrix,
                        alternative_scale,
                        current_scale,
                        changes,
                    )
                else:
                    p_type = np.nan

                # Refine onset & offset
                bound_tol = round(config_params["P Module"]["P_bound_tol"] * freq)
                search_range_onset_start = max(
                    window_start, estimated_onset - bound_tol
                )
                search_range_onset_end = estimated_onset
                search_range_offset_start = estimated_offset
                search_range_offset_end = (
                    min(estimated_offset + bound_tol, window_end) + 1
                )

                if estimated_onset:
                    onset_matrix = wavelet_matrix[
                        search_range_onset_start:search_range_onset_end,
                        current_scale,
                    ]
                    refined_onset = (
                        search_onset(estimated_onset, onset_matrix, onset_percision) - 1
                    )

                if estimated_offset:
                    offset_matrix = wavelet_matrix[
                        search_range_offset_start:search_range_offset_end,
                        current_scale,
                    ]
                    refined_offset = (
                        search_offset(estimated_offset, offset_matrix, offset_percision)
                        + 1
                    )

                # Reset variable values (A)
                if (not p_wave_peaks and "reset_refinement" in changes) or (
                    not refined_onset or not refined_offset
                ):
                    refined_onset, refined_offset, p_wave_peaks = [], [], []
                max_pre_peak_positions, max_post_peak_positions = [], []
                min_pre_peak_position, min_post_peak_positions = [], []
                max_position, min_position = [], []
            else:
                max_position, min_position = [], []

            # Reset variable values (B)
            refined_onset = to_nan(refined_onset)
            refined_offset = to_nan(refined_offset)
            p_wave_peaks = to_nan(p_wave_peaks)
            estimated_offset = to_nan(estimated_offset)
            estimated_onset = to_nan(estimated_onset)
            secondary_peak = to_nan(secondary_peak)
            if not p_wave_peaks:
                p_type = np.nan
                current_scale = np.nan

            # Updating the locations dictionary
            locations["P_Wave_Onset"][i] = refined_onset + sample_excerpt[0] - 1
            locations["P_Wave_Offset"][i] = refined_offset + sample_excerpt[0] - 1
            locations["P_Wave_Peak"][i] = p_wave_peaks + sample_excerpt[0]
            locations["P_Wave_Prime"][i] = secondary_peak + sample_excerpt[0] - 1
            locations["P_Wave_Scale"][i] = current_scale
            locations["P_Wave_Type"][i] = p_type

            # Updating the ECG analysis arrays
            estimated_onset_array.append(estimated_onset)
            estimated_offset_array.append(estimated_offset)
            p_wave_peaks, secondary_peak = [], []
            estimated_onset, estimated_offset = [], []
            refined_onset, refined_offset = [], []

    return locations, estimated_onset_array, estimated_offset_array
