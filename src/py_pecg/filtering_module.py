import numpy as np
from scipy.signal import firls, lfilter


def interp(idata: np.ndarray, r: int, n: int, cutoff: float) -> np.ndarray:
    """Resample data at a higher rate using lowpass interpolation. Resamples the
    sequence in vector X at R times the original sample rate. The resulting resampled
    vector Y is R times longer, LENGTH(Y) = R*LENGTH(X). A symmetric filter, B, allows
    the original data to pass through unchanged and interpolates between so that the
    mean square error between them and their ideal values is minimized.

    Args:
        idata (np.ndarray): The target signal designated for interpolation.
        r (int): The interpolation factor of the resulting resampled vector.
        n (int): Half the number of original sample values used to perform the
        interpolation. For best results, use N no larger than 10. The length of
         B is 2*N*R+1.
        cutoff (float): Cutoff Frequency. The signal is assumed to be band
                        limited with cutoff frequency 0 < CUTOFF <= 1.0.

    Raises:
        TypeError: Type validaiton for r,n and cutoff.
        ValueError: Value validation for cutoff.

    Returns:
        b (np.ndarray): returns the coefficients of the interpolation filter B.
        odata (np.ndarray): returns the interpolated version of the target signal.

    Notes:
        i: Filter a fabricated section of data first (match initial values and first
        derivatives by rotating the first data points by 180 degrees) to get
        guess of good initial conditions. Filter length is 2*r*n+1 so need that
        many points; can't duplicate first point or guarantee a zero slope at
        beginning of sequence.
        ii: Make sure right hand points of data have been correctly interpolated and
        get rid of transients by again matching end values and derivatives of the
        original data.

    Credits:
        Copyright 1988-2022 The MathWorks, Inc.
        "Programs for Digital Signal Processing", IEEE Press
        John Wiley & Sons, 1979, Chap. 8.1.
    """

    def designInterpFilt(r: int, n: int, alpha: float) -> np.ndarray:
        """This function designs a linear-phase FIR filter of type I for interpolation.

        Args:
            r (int): The interpolation factor of the resulting resampled vector.
            n (int): Half the number of original sample values used to perform the
            interpolation. alpha (float): Variation of the cutoff frequency
            specification.

        Returns:
            b (TYPE): The resultant interpolation filter coefficients.
        """
        # filter specification (frequency and magnitude)
        if alpha == 1:
            M = np.array([r, r, 0, 0])
            F = np.array([0, 1 / (2 * r), 1 / (2 * r), 0.5])
        else:
            Nband = int((np.floor(0.5 * r)))
            M = np.concatenate((np.array([r, r]), np.zeros((1, 2 * Nband)).flatten()))
            a2r = alpha / 2 / r
            F = np.zeros((1, 2 * Nband + 2)).flatten()
            F[1] = a2r
            k = 0
            for i in range(2, 2 * Nband + 2, 2):
                k = k + 1
                F[i] = k / r - a2r
                F[i + 1] = k / r + a2r
            if F[2 * Nband + 1] > 0.5:
                F[2 * Nband + 1] = 0.5
            # design filter, cast to proper data type and convert it to a column vector
            b = (
                firls(2 * r * n + 1, 2 * F, M).astype(alpha.dtype).T
            )  # firls returns a row vector
            return b

    # Argument Validation
    if cutoff == 0:
        cutoff = 0.5

    if isinstance(idata, list):
        raise TypeError("Input data must be a numpy array, not a list.")
    if not isinstance(idata, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if idata.size == 0:
        raise ValueError("Input is an empty array.")
    if idata.squeeze().ndim > 1:
        raise ValueError("Input is a higher-dimensional array, not a matrix.")

    if not isinstance(r, (int, np.integer)):
        raise TypeError("r must be a finite integer scalar.")
    if r < 1 or not np.isfinite(r):
        raise ValueError("r must be >= 1 and finite.")

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be a finite integer scalar.")
    if n < 1 or not np.isfinite(n):
        raise ValueError("n must be >= 1 and finite.")

    if not isinstance(cutoff, (int, float)):
        raise TypeError("cutoff must be a finite scalar.")
    if cutoff <= 0 or cutoff > 1 or not np.isfinite(cutoff):
        raise ValueError("cutoff must be > 0, <= 1, and finite.")

    # Convert all input parameters to the same data type as the input signal
    if idata.dtype == np.float32:
        r = np.array(r, dtype=np.float32)
        n = np.array(n, dtype=np.float32)
        cutoff = np.array(cutoff, dtype=np.float32)
    else:
        r = np.array(r, dtype=np.float64)
        n = np.array(n, dtype=np.float64)
        cutoff = np.array(cutoff, dtype=np.float64)

    # Determine if the input is a row vector
    idataIsRow = idata.ndim == 1 and idata.shape[0] == idata.size
    if idataIsRow:
        xCol = np.reshape(idata, (-1, 1)).flatten()
    else:
        xCol = idata

    # Check if filter length before interpolation is smaller than input length
    Lx = len(xCol)
    if not (2 * n + 1 < Lx):
        raise ValueError("signal:interp:InvalidDimensions: 2*n + 1 must be < Lx")

    # constant values used for indexing and setting vector sizes
    RL = int(r * Lx)
    RN = int(r * n)

    # Design the linear-phase FIR filter of type I
    b = designInterpFilt(r, n, cutoff)

    # Use the filter B to perform the interpolation
    odt = np.zeros_like(xCol[0]).flatten()
    yCol = np.zeros((RL, 1), dtype=odt.dtype).flatten()
    temp_indices = np.arange(0, RL, int(r))  # Create indices 0, r, 2r, ..., up to RL-1
    yCol[temp_indices] = xCol

    # Filter a fabricated section of data
    od = np.zeros((2 * RN, 1)).flatten()
    temp_indices_2 = np.arange(0, RN * 2, int(r))
    temp_indices_2_B = np.arange(2 * int(n), 0, -1)
    od[temp_indices_2] = 2 * xCol[0] - xCol[temp_indices_2_B]
    zi = lfilter(np.flip(b), 1, np.flip(od))
    zi = np.flip(zi)

    yCol, zf = lfilter(b, 1, yCol, zi=zi)
    temp_indices_3 = np.arange(0, (Lx - int(n)) * int(r))
    temp_indices_3_B = np.arange(RN, RL)
    yCol[temp_indices_3] = yCol[temp_indices_3_B]

    # Make sure right hand points of data have been correctly interpolated
    od = np.zeros((2 * RN, 1)).flatten()
    temp_indices_4 = np.arange(0, RN * 2, int(r))
    temp_indices_4_B = np.arange(Lx - 2, Lx - 2 * int(n) - 2, -1)
    od[temp_indices_4] = 2 * xCol[Lx - 1] - (xCol[temp_indices_4_B])
    od, _ = lfilter(b, 1, od, zi=zf)

    temp_indices_5 = np.arange(RL - RN, RL)
    temp_indices_5B = np.arange(0, RN)
    yCol[temp_indices_5] = od[temp_indices_5B]
    # yCol[RL-RN+1:RL,1] = od[1:RN,1];

    # Convert output to be a row vector if the input is a row
    if idataIsRow:
        odata = np.reshape(yCol, (1, RL))
    else:
        odata = yCol

    return [odata.flatten(), b]


"""
Credits:
    Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
    Ana Paula Rocha, Pablo Laguna
    Original Publication: A wavelet-based ECG delineator:
    evaluation on standard databases
    Original DOI: 10.1109/TBME.2003.821031.
"""


def quadratic_splines_filterbank(fs: float, messages: dict):
    """Obtains the quadratic splines wavelet filterbank filters from scale 1 to 4 as a
    function of the sampling frequency, in order to use filters with similar analog
    frequency behaviour for diferent sampling frecuencies.

    Args:
        fs (float): Sampling frequency of the target signal.
        messages (dict): A dictionary of in-code configurations.

    Returns:
        quadratic_filter_array (list): A list of numpy arrays representing filters
        at different scales.
        messages (dict): Dictionary containing information relevant to the algorithm.

    Note:
        Initializing algorithme a trous (Decimation)
        Transfer Functions:
        Q1(w) = G(w);
        Q2(w) = H(w)G(2w);
        Q3(w) = H(w)H(2w)G(4w);
        Q4(w) = H(w)H(2w)H(4w)G(8w);
        Q5(w) = H(w)H(2w)H(4w)H(8w)G(16w);
    """
    if fs < 0:
        raise ValueError("Sampling Frequency must be positive, and non-zero.")

    if np.size(fs) == 0:
        raise ValueError("Insert Sampling Frequency!")

    if not isinstance(fs, (int, float)) or isinstance(fs, bool):
        raise ValueError(
            "Input must be a scalar integer or float, not "
            "an array or higher-dimensional object."
        )

    if not ((fs == 500) or (fs == 1000) or (fs == 360) or (fs == 200)):
        raise ValueError("Sampling Frequency must be either 200, 360, 500 or 1000.")
        messages["errors"].append(
            "There are no wavelets designed for this sampling frequency."
        )
        messages["errors_desc"].append(
            "Default wavelet filters not defined for thia samplig frequency."
            " Use filter design instead."
        )
        messages["status"] = 0

    messages["status"] = 1

    def create_ordered_filter(base_filter, zeros_count):
        return np.kron(base_filter, np.array([1] + [0] * zeros_count))

    def apply_interpolation(quadratic_filter_array, interp_params):
        for i, (filt, params) in enumerate(zip(quadratic_filter_array, interp_params)):
            if params is False:  # Skip interpolation for this filter if params is False
                continue
            step, zero_count, fill_value = params
            quadratic_filter_array[i], _ = interp(
                np.concatenate(([0], filt, [0])), step, zero_count, fill_value
            )
        return quadratic_filter_array

    hpf_ord_1 = 0.125 * np.array([1, 3, 3, 1])
    lpf_ord_1 = 2 * np.array([1, -1])

    zeros_counts = [1, 3, 7, 15]
    hpf_filters = [hpf_ord_1] + [
        create_ordered_filter(hpf_ord_1, zeros) for zeros in zeros_counts
    ]
    lpf_filters = [lpf_ord_1] + [
        create_ordered_filter(lpf_ord_1, zeros) for zeros in zeros_counts
    ]

    quadratic_filter_array = [
        lpf_ord_1,  # quadratic_filt_1
        np.convolve(hpf_filters[0], lpf_filters[1]),  # quadratic_filt_2
        np.convolve(
            np.convolve(hpf_filters[0], hpf_filters[1]), lpf_filters[2]
        ),  # quadratic_filt_3
        np.convolve(
            np.convolve(np.convolve(hpf_filters[0], hpf_filters[1]), hpf_filters[2]),
            lpf_filters[3],
        ),  # quadratic_filt_4
        np.convolve(
            np.convolve(
                np.convolve(
                    np.convolve(hpf_filters[0], hpf_filters[1]), hpf_filters[2]
                ),
                hpf_filters[3],
            ),
            lpf_filters[4],
        ),  # quadratic_filt_5
    ]
    # Filter Truncation
    filter_lengths = [max(np.argwhere(filt != 0)) for filt in quadratic_filter_array]
    for i, filt in enumerate(quadratic_filter_array):
        length = filter_lengths[i][0]
        quadratic_filter_array[i] = np.transpose(filt[: length + 1])

    if fs == 500:
        # Interpolate by 2
        interp_params = [(2, 1, 0), (2, 3, 0), (2, 7, 0), (2, 7, 0), (2, 7, 0)]
        quadratic_filter_array = apply_interpolation(
            quadratic_filter_array, interp_params
        )
        for i in range(0, len(quadratic_filter_array)):
            quadratic_filter_array[i] = quadratic_filter_array[i][1:-2]

    elif fs == 1000:
        # Interpolate by 4
        interp_params = [(4, 1, 0), (4, 3, 0), (4, 7, 0), (4, 7, 0), (4, 7, 0)]
        quadratic_filter_array = apply_interpolation(
            quadratic_filter_array, interp_params
        )
        for i in range(0, len(quadratic_filter_array)):
            quadratic_filter_array[i] = quadratic_filter_array[i][1:-4]

    elif fs == 200:
        # Interpolate by 4
        interp_params = [
            False,
            (4, 3, 0.4),
            (4, 7, 0.4),
            (4, 7, 0.4),
            (4, 7, 0.4),
        ]
        quadratic_filter_array = apply_interpolation(
            quadratic_filter_array, interp_params
        )
        for i in range(0, len(quadratic_filter_array)):
            quadratic_filter_array[i] = quadratic_filter_array[i][1:-4]

        # Downsample by 5
        quadratic_filter_array[0] = 1.1 * np.array([5 / 4, -5 / 4])
        for i, ind in enumerate([False, 3, 4, 1, 1]):
            if i != 0:
                quadratic_filter_array[i] = quadratic_filter_array[i][ind::5]

    elif fs == 360:
        # Interpolate by 6
        interp_params = [(6, 1, 0), (6, 3, 0), (6, 7, 0), (6, 7, 0), (6, 7, 0)]
        quadratic_filter_array = apply_interpolation(
            quadratic_filter_array, interp_params
        )
        for i in range(0, len(quadratic_filter_array)):
            quadratic_filter_array[i] = quadratic_filter_array[i][1:-6]

        # Downsample by 5
        for i, ind in enumerate([3, 0, 4, 2, 2]):
            quadratic_filter_array[i] = quadratic_filter_array[i][ind::5]

        # Interpolate by 6 again
        interp_params = [(6, 1, 0), (6, 3, 0), (6, 7, 0), (6, 7, 0), (6, 7, 0)]
        quadratic_filter_array = apply_interpolation(
            quadratic_filter_array, interp_params
        )
        for i in range(0, len(quadratic_filter_array)):
            quadratic_filter_array[i] = quadratic_filter_array[i][1:-6]

        # Final Downsample by 5
        for i, ind in enumerate([1, 4, 3, 3, 3]):
            quadratic_filter_array[i] = quadratic_filter_array[i][ind::5]

    return quadratic_filter_array, messages


def wavelet_transform(
    signal: np.ndarray,
    quadratic_filter_array: np.ndarray,
) -> np.ndarray:
    """Calculates the wavelet transform of a signal using quadratic spline wavelet. It
    calculates wavelets in scales of 1 to 4.

    Args:
        signal (np.ndarray): The target signal designated for filtering.
        quadratic_filter_array (np.ndarray): Qquadratic spline filter bank.

    Returns:
        wavelet_transform_matrix (np.ndarray): Resultant Wavelet Transform Matrix.
    """
    if signal.size == 0:
        raise ValueError("Input is an empty array.")
    if not isinstance(quadratic_filter_array, list) or any(
        arr.size == 0 for arr in quadratic_filter_array
    ):
        raise ValueError("quadratic_filter_array should be a list of 1D numpy arrays.")
    if signal.squeeze().ndim > 1 or any(
        not isinstance(arr, np.ndarray) or arr.ndim != 1
        for arr in quadratic_filter_array
    ):
        raise ValueError("Input is a higher-dimensional array, not a matrix.")

    wavelet_transform_matrix = np.zeros((len(signal), 5))
    for i, filt in enumerate(quadratic_filter_array):
        wavelet_transform_matrix[:, i] = np.transpose(lfilter(filt, 1, signal))
    return wavelet_transform_matrix


def create_filter(messages: dict) -> dict:
    """Creates quadratic spline filters using a variation on Mallat's Algorithm,
    algorithme à trous.

    Args:
        messages (dict): Dictionary containing information relevant to the algorithm.

    Returns:
        quadratic_filter_dictionary (dict): A dictionary relevant to the algorithm
        filter-bank.
    """
    quadratic_filter_array, messages = quadratic_splines_filterbank(
        messages["setup"]["wavedet"]["freq"], messages
    )
    lengths = [len(q) for q in quadratic_filter_array]
    decimation_values = [int(np.floor((length - 1) / 2)) for length in lengths]
    quadratic_filter_dictionary = {
        "filters": quadratic_filter_array,
        "lengths": lengths,
        "d_values": decimation_values,
    }
    return quadratic_filter_dictionary


def filter_segment_of_signal(
    quadratic_filter_dictionary: dict,
    current_samp: int,
    initial_samp: int,
    num_samps: int,
    segment_boundaries: list,
    signal_segment: np.ndarray,
    messages: dict,
):
    """Uses a Wavelet Transform on individual ECG signal excerpts, after Generating
    quadratic-spline filter banks.

    Args:
        quadratic_filter_dictionary (dict): A dictionary relevant to the algorithm
        filter-bank.
        current_samp (int): Current sample analyzed within the ECG signal.
        initial_samp (int): Initial sample analyzed within the ECG signal.
        num_samps (int): Number of samples per excerpt for the filter bank
        corresponding to 2^16 samples at sf=250.
        segment_boundaries (list): First and last indices analyzed within the ECG
        signal.
        signal_segment (np.ndarray): The ECG signal designated for analysis.
        messages (dict): Dictionary containing information relevant to the algorithm.

    Returns:
        signal_segment (np.ndarray): The ECG signal designated for analysis.
        current_samp (int): Updated sample analyzed within the ECG signal.
        updated_samp (int): Updated current sample analyzed within the ECG signal.
        synchronized_wavelet_matrix (np.ndarray): Wavelet Transform Matrix, relevant
        to the current segment of the signal.
        threshold_matrix (np.ndarray): QRS detection threshold matrix.
        end_samp (int): Updated final sample analyzed within the ECG signal.
        initial_samp (int): Updated initial sample analyzed within the ECG signal.

    NOTE:
        First l5-1 samples are not correctly filtered (border effect)
        Last d5 samples are discarded in order to alineate all the
        filtered signals, taking into acount the filter delays.
    """

    def process_filters(quadratic_filter_dictionary):
        required_keys = {"filters", "lengths", "d_values"}
        keys_in_dict = set(quadratic_filter_dictionary.keys())
        missing_keys = required_keys - keys_in_dict
        extra_keys = keys_in_dict - required_keys
        if missing_keys:
            raise ValueError(
                f"quadratic_filter_dictionary is missing keys: {missing_keys}"
            )
        if extra_keys:
            raise ValueError(
                f"quadratic_filter_dictionary contains unexpected keys: {extra_keys}"
            )
        quadratic_filter_array, lengths, decimation_values = [
            quadratic_filter_dictionary[key] for key in required_keys
        ]
        if not quadratic_filter_array or not lengths or not decimation_values:
            raise ValueError(
                "quadratic_filter_dictionary contains empty values for "
                "'filters', 'lengths', or 'd_values'."
            )
        return

    # Validation
    process_filters(quadratic_filter_dictionary)
    for component in (current_samp, initial_samp, num_samps):
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
            "segment_boundaries must be a non-empty list of integers or floats."
        )

    # Filter-bank extraction
    quadratic_filter_array, lengths, decimation_values = [
        quadratic_filter_dictionary[key]
        for key in [
            "filters",
            "lengths",
            "d_values",
        ]
    ]

    # Last sample excerpt analyzed
    updated_samp = current_samp
    end_samp = min(initial_samp + num_samps - 1, segment_boundaries[1])
    initial_samp = round(initial_samp)
    end_samp = round(end_samp)
    num_samps = round(num_samps)

    # Filter bank - Wavelet Matrix construction and parameters setting
    l5 = lengths[-1]
    wavelet_transform_matrix = wavelet_transform(signal_segment, quadratic_filter_array)
    wavelet_transform_matrix = wavelet_transform_matrix[l5 - 1 :, 0:5]

    # Synchronizing filtered signals at different scales
    d5 = decimation_values[-1]
    synchronized_wavelet_matrix = np.zeros((len(wavelet_transform_matrix) - d5, 5))
    for i, d in enumerate(decimation_values):
        synchronized_wavelet_matrix[:, i] = wavelet_transform_matrix[
            d : d + len(wavelet_transform_matrix) - d5, i
        ]

    # Remove "incorrect" samples (see NOTE)
    current_samp = np.arange(initial_samp + l5 - 1, end_samp - d5 + 1)

    # Evaluate quality of the signal
    swt = synchronized_wavelet_matrix**2
    threshold_matrix = (
        0.5
        * np.sqrt(np.median(swt, axis=0))
        * messages["setup"]["wavedet"]["QRS_detection_thr"]
    )
    threshold_matrix[3, 0] = threshold_matrix[3, 0] * 2
    signal_segment = signal_segment[l5 - 1 : len(signal_segment) - d5]

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
