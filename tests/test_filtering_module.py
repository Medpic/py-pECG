import json
from itertools import product

import numpy as np
import pytest

from py_pecg.filtering_module import (
    create_filter,
    filter_segment_of_signal,
    interp,
    quadratic_splines_filterbank,
    wavelet_transform,
)


@pytest.fixture
def load_quadratic_filter_array():
    def load_for_frequency(frequency):
        filename = (
            f"./test_filtering_module_data/quadratic_filter_array_"
            f"{frequency}_Hz.npz"
        )
        loaded_file = np.load(filename)
        return [loaded_file[f"arr_{i}"] for i in range(len(loaded_file.files))]

    return load_for_frequency


@pytest.fixture
def load_quadratic_filter_dictionary():
    def load_for_frequency(frequency):
        with open(
            f"./test_filtering_module_data/quadratic_filter_dictionary_"
            f"{frequency}_Hz.json",
            "r",
        ) as file:
            loaded_file = json.load(file)
            loaded_quadratic_filter_list = loaded_file["filters"]
            loaded_quadratic_filter_lengths = loaded_file["lengths"]
            loaded_filter_decimation_values = loaded_file["d_values"]
        return (
            loaded_quadratic_filter_list,
            loaded_quadratic_filter_lengths,
            loaded_filter_decimation_values,
        )

    return load_for_frequency


@pytest.fixture
def lists_of_arrays_equal():
    def _compare(list1, list2):
        if len(list1) != len(list2):
            return False
        for arr1, arr2 in zip(list1, list2):
            if not np.array_equal(arr1, arr2):
                return False
        return True

    return _compare


def test_interp():
    def interpolation_factor_test(signal):
        # Interpolation Factor (r) Test for valid LPF length (n) values
        # (2*n + 1 < len(signal))
        for r in range(1, 5):
            for n in range(1, int(np.floor((len(signal) - 1) // 2 + 1)) - 1):
                interpolated_signal, _ = interp(signal, r, n, 0.5)
                assert len(interpolated_signal) == r * len(signal)
                assert interpolated_signal.ndim == 1

    def low_pass_filter_length_test(signal):
        # Invalidating cases in which (2*n + 1 < len(signal))
        for r, n in product(range(1, 5), range(1, 5)):
            if n in range(1, int(np.floor((len(signal) - 1) // 2 + 1)) - 1):
                continue
            else:
                with pytest.raises(ValueError):
                    interp(signal, r, n, 0.5)

    # Valid Cases
    # Row Vector and Column Vector Input
    for signal_length in range(5, 20, 5):
        interpolation_factor_test(np.linspace(0, signal_length, signal_length + 1))
        interpolation_factor_test(
            np.transpose(np.linspace(0, signal_length, signal_length + 1))
        )

    # Invalid Cases
    # Invalid Signal Input
    with pytest.raises(ValueError):
        interp([], 2, 1, 0.5)
    with pytest.raises(TypeError):
        interp(1, 2, 1, 0.5)
    with pytest.raises(ValueError):
        interp(np.array([]), 2, 1, 0.5)
    with pytest.raises(ValueError):
        interp(np.array([[2, 2], [2, 2]]), 2, 1, 0.5)

    # Invalid Interpolation Factor
    with pytest.raises(ValueError):
        interp(np.linspace(0, 10, 11), 0, 1, 0.5)
    with pytest.raises(ValueError):
        interp(np.linspace(0, 10, 11), -2, 1, 0.5)
    with pytest.raises(TypeError):
        interp(np.linspace(0, 10, 11), 0.3, 1, 0.5)

    # Invalid LPF Length
    low_pass_filter_length_test(np.linspace(0, 10, 11))

    # Invalid Cutoff Frequency
    with pytest.raises(ValueError):
        interp(np.linspace(0, 10, 11), 2, 1, 2)
    with pytest.raises(ValueError):
        interp(np.linspace(0, 10, 11), 2, 1, -1)
    return


def test_quadratic_splines_filterbank(
    load_quadratic_filter_array, lists_of_arrays_equal
):
    def quadratic_splines_subtest(frequency, lists_of_arrays_equal):
        result = quadratic_splines_filterbank(frequency)
        ground_truth = load_quadratic_filter_array(frequency)

        # Valid Cases
        assert isinstance(result, list)
        assert len(result) == 5
        assert isinstance(result[0], np.ndarray)
        assert lists_of_arrays_equal(ground_truth, result)

    for freq in [500, 360, 1000, 200]:
        quadratic_splines_subtest(freq, lists_of_arrays_equal)

    # Invalid Cases
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(-10)
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(123)
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(np.array([1, 1]))
    return


def test_wavelet_transform(load_quadratic_filter_array):
    def wavelet_subtest(signal, quadratic_filter_array):
        wavelet_transform_matrix = wavelet_transform(signal, quadratic_filter_array)
        assert wavelet_transform_matrix.shape == (
            len(signal),
            len(quadratic_filter_array),
        )
        return wavelet_transform_matrix

    # Loading the Filters
    quadratic_filter_array_500_Hz = load_quadratic_filter_array(500)

    # Valid Cases
    assert (
        np.all(wavelet_subtest(np.zeros((10, 1)), quadratic_filter_array_500_Hz)) == 0
    )  # Zero Signal
    assert not np.all(
        wavelet_subtest(np.ones((10, 1)), quadratic_filter_array_500_Hz)[:, 0] == 0
    )  # Const. Signal, Low Freq.
    assert np.allclose(
        wavelet_subtest(np.ones((10, 1)), quadratic_filter_array_500_Hz)[:, 1:],
        0,
        atol=1e-1,
    )  # Const. Signal, High Freq.
    assert np.any(
        wavelet_subtest(
            np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500)),
            quadratic_filter_array_500_Hz,
        )
        != 0
    )  # Sine Signal

    # Invalid Cases
    with pytest.raises(ValueError):
        wavelet_transform(np.zeros((10, 10)), quadratic_filter_array_500_Hz)
    with pytest.raises(ValueError):
        wavelet_transform(np.array([]), quadratic_filter_array_500_Hz)
    with pytest.raises(ValueError):
        wavelet_transform(np.ones((10, 1)), 1)
    with pytest.raises(ValueError):
        wavelet_transform(np.ones((10, 1)), np.array((10, 10, 10)))
    return


def test_create_filter(load_quadratic_filter_dictionary, lists_of_arrays_equal):
    for freq in [500, 360, 1000, 200]:
        (
            quadratic_filter_list,
            quadratic_filter_lengths,
            filter_decimation_values,
        ) = create_filter(freq)
        (
            loaded_quadratic_filter_list,
            loaded_quadratic_filter_lengths,
            loaded_filter_decimation_values,
        ) = load_quadratic_filter_dictionary(freq)
        assert lists_of_arrays_equal(
            loaded_quadratic_filter_list, quadratic_filter_list
        )
        assert lists_of_arrays_equal(
            loaded_quadratic_filter_lengths, quadratic_filter_lengths
        )
        assert lists_of_arrays_equal(
            loaded_filter_decimation_values, filter_decimation_values
        )
    return


def test_filter_segment_of_signal():
    # Generating Test Data
    (
        quadratic_filter_list,
        quadratic_filter_lengths,
        filter_decimation_values,
    ) = create_filter(500)
    signal_length = 5000
    num_samps = 500
    current_samp = 0
    signal = np.random.random(signal_length)

    for current_samp in range(0, signal_length, num_samps):
        end = min(current_samp + num_samps, signal_length)
        signal_segment = signal[current_samp:end]
        initial_samp = current_samp
        end_samp = end - 1
        segment_boundaries = [initial_samp, end_samp]

        # Invalid Cases
        test_cases = {
            "quadratic_filter_list": [[], np.array([1, 1]), np.array([])],
            "quadratic_filter_lengths": [
                -1,
                [],
                np.array([1, 1]),
                np.array([]),
            ],
            "filter_decimation_values": [[], np.array([1, 1]), np.array([])],
            "current_samp": [-1, 1.1, np.array([1, 1])],
            "initial_samp": [-1, 1.1, np.array([1, 1])],
            "num_samps": [-1, 1.1, np.array([1, 1])],
            "segment_boundaries": [1, [1.1, 2], [], np.array([])],
            "signal_segment": [np.array([])],
        }

        for param_name, invalid_values in test_cases.items():
            for invalid_value in invalid_values:
                with pytest.raises(ValueError):
                    filter_segment_of_signal(
                        (
                            invalid_value
                            if param_name == "quadratic_filter_list"
                            else quadratic_filter_list
                        ),
                        (
                            invalid_value
                            if param_name == "quadratic_filter_lengths"
                            else quadratic_filter_lengths[-1]
                        ),
                        (
                            invalid_value
                            if param_name == "filter_decimation_values"
                            else filter_decimation_values
                        ),
                        (
                            invalid_value
                            if param_name == "current_samp"
                            else current_samp
                        ),
                        (
                            invalid_value
                            if param_name == "initial_samp"
                            else initial_samp
                        ),
                        (invalid_value if param_name == "num_samps" else num_samps),
                        (
                            invalid_value
                            if param_name == "segment_boundaries"
                            else segment_boundaries
                        ),
                        (
                            invalid_value
                            if param_name == "signal_segment"
                            else signal_segment
                        ),
                    )

        # Valid Cases
        (
            processed_signal_segment,
            updated_current_samp,
            updated_samp,
            synchronized_wavelet_matrix,
            threshold_matrix,
            updated_end_samp,
            updated_initial_samp,
        ) = filter_segment_of_signal(
            quadratic_filter_list,
            quadratic_filter_lengths[-1],
            filter_decimation_values,
            current_samp,
            initial_samp,
            num_samps,
            segment_boundaries,
            signal_segment,
        )

        assert isinstance(
            processed_signal_segment, np.ndarray
        )  # Processed Signal Segment
        assert isinstance(updated_current_samp, np.ndarray)  # Updated Current Sampling
        # Updated Sample
        assert isinstance(updated_samp, int)
        assert segment_boundaries[0] <= updated_samp <= segment_boundaries[1] + 1
        # Synchronized Wavelet Matrix
        assert isinstance(synchronized_wavelet_matrix, np.ndarray)
        assert synchronized_wavelet_matrix.shape[1] == 5
        # Threshold Matrix
        assert isinstance(threshold_matrix, np.ndarray)
        assert threshold_matrix.shape[1] == 5
        # Updated End Sample
        assert isinstance(updated_end_samp, int)
        assert updated_end_samp <= segment_boundaries[1]
        # Updated Initial Sample
        assert isinstance(updated_initial_samp, int)
        assert updated_initial_samp >= segment_boundaries[0]

    return


pytest.main(
    [
        "--cov=filtering_module",
        "--cov-report=term-missing",
        "--disable-warnings",
        "test_filtering_module.py",
    ]
)
