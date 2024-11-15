import json
from itertools import product

import numpy as np
import pytest

from filtering_module import (
    create_filter,
    filter_segment_of_signal,
    interp,
    quadratic_splines_filterbank,
    wavelet_transform,
)


@pytest.fixture
def messages():
    # Standard messages array for the algorithm
    with open("test_filtering_module_messages.json", "r") as file:
        messages = json.load(file)
    return messages


@pytest.fixture
def load_quadratic_filter_array():
    def load_for_frequency(frequency):
        filename = f"quadratic_filter_array_{frequency}_Hz.npz"
        loaded_file = np.load(filename)
        return [loaded_file[f"arr_{i}"] for i in range(len(loaded_file.files))]

    return load_for_frequency


@pytest.fixture
def load_quadratic_filter_dictionary():
    def load_for_frequency(frequency):
        with open(
            f"quadratic_filter_dictionary_{frequency}_Hz.json", "r"
        ) as file:
            loaded_file = json.load(file)
        return loaded_file

    return load_for_frequency


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
        interpolation_factor_test(
            np.linspace(0, signal_length, signal_length + 1)
        )
        interpolation_factor_test(
            np.transpose(np.linspace(0, signal_length, signal_length + 1))
        )

    # Invalid Cases
    # Invalid Signal Input
    with pytest.raises(TypeError):
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


def test_quadratic_splines_filterbank(load_quadratic_filter_array, messages):
    def quadratic_splines_subtest(frequency, messages):
        def lists_of_arrays_equal(list1, list2):
            if len(list1) != len(list2):
                return False
            for arr1, arr2 in zip(list1, list2):
                if not np.array_equal(arr1, arr2):
                    return False
            return True

        # Loading the Filters
        result, messages = quadratic_splines_filterbank(frequency, messages)
        ground_truth = load_quadratic_filter_array(frequency)

        # Valid Cases
        assert isinstance(result, list)
        assert len(result) == 5
        assert isinstance(result[0], np.ndarray)
        assert messages["status"] == 1
        assert lists_of_arrays_equal(ground_truth, result)

    for freq in [500, 360, 1000, 200]:
        quadratic_splines_subtest(freq, messages)

    # Invalid Cases
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(-10, messages)
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(123, messages)
    with pytest.raises(ValueError):
        quadratic_splines_filterbank(np.array([1, 1]), messages)
    return


def test_wavelet_transform(load_quadratic_filter_array):
    def wavelet_subtest(signal, quadratic_filter_array):
        wavelet_transform_matrix = wavelet_transform(
            signal, quadratic_filter_array
        )
        assert wavelet_transform_matrix.shape == (
            len(signal),
            len(quadratic_filter_array),
        )
        return wavelet_transform_matrix

    # Loading the Filters
    quadratic_filter_array_500_Hz = load_quadratic_filter_array(500)

    # Valid Cases
    assert (
        np.all(
            wavelet_subtest(np.zeros((10, 1)), quadratic_filter_array_500_Hz)
        )
        == 0
    )  # Zero Signal
    assert not np.all(
        wavelet_subtest(np.ones((10, 1)), quadratic_filter_array_500_Hz)[:, 0]
        == 0
    )  # Const. Signal, Low Freq.
    assert np.allclose(
        wavelet_subtest(np.ones((10, 1)), quadratic_filter_array_500_Hz)[
            :, 1:
        ],
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
        wavelet_transform(np.ones((10, 1)), np.array([]))
    with pytest.raises(ValueError):
        wavelet_transform(np.ones((10, 1)), np.zeros((10, 10, 10)))
    return


def test_create_filter(load_quadratic_filter_dictionary, messages):
    def compare_dicts_with_arrays(dict1, dict2):
        def convert_arrays_to_lists(data):
            if isinstance(data, dict):
                return {
                    key: convert_arrays_to_lists(value)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                return [convert_arrays_to_lists(item) for item in data]
            elif isinstance(data, np.ndarray):
                return data.tolist()
            else:
                return data

        dict1_converted = convert_arrays_to_lists(dict1)
        dict2_converted = convert_arrays_to_lists(dict2)
        return dict1_converted == dict2_converted

    # Comparing output to Ground Truth for each frequency.
    for freq in [500, 360, 1000, 200]:
        messages["setup"]["wavedet"]["freq"] = freq
        quadratic_filter_dictionary = create_filter(messages)
        ground_truth = load_quadratic_filter_dictionary(freq)
        assert compare_dicts_with_arrays(
            ground_truth, quadratic_filter_dictionary
        )
    return


def test_filter_segment_of_signal(messages):
    # Generating Test Data
    messages["setup"]["wavedet"]["freq"] = 500
    quadratic_filter_dictionary = create_filter(messages)
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
            "quadratic_filter_dictionary": {},
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
                            if param_name == "quadratic_filter_dictionary"
                            else quadratic_filter_dictionary
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
                        (
                            invalid_value
                            if param_name == "num_samps"
                            else num_samps
                        ),
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
                        messages,
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
            quadratic_filter_dictionary,
            current_samp,
            initial_samp,
            num_samps,
            segment_boundaries,
            signal_segment,
            messages,
        )

        assert isinstance(
            processed_signal_segment, np.ndarray
        )  # Processed Signal Segment
        assert isinstance(
            updated_current_samp, np.ndarray
        )  # Updated Current Sampling
        # Updated Sample
        assert isinstance(updated_samp, int)
        assert (
            segment_boundaries[0] <= updated_samp <= segment_boundaries[1] + 1
        )
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
