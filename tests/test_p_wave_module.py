import json

import numpy as np
import pytest

from ..src.py_pecg.p_wave_module import (
    detect_p_wave_features,
    find_p_wave,
    find_secondary_extrema,
    refine_complex_p_wave,
    refine_simple_p_wave,
)


@pytest.fixture
def load_numpy_array():
    def load_for_name(name):
        loaded_file = np.load(f"./test_p_wave_module_data/{name}.npz")
        if len(loaded_file.files) == 1:
            return loaded_file[loaded_file.files[0]]
        raise ValueError(f"{name}.npz contains multiple arrays. Expected only one.")

    return load_for_name


@pytest.fixture
def load_dictionary():
    def load_for_name(name):
        with open(f"./test_p_wave_module_data/{name}.json", "r") as file:
            loaded_file = json.load(file)

        def convert_to_numpy(obj):
            if isinstance(obj, list):
                return np.array(obj)
            if isinstance(obj, dict):
                return {key: convert_to_numpy(value) for key, value in obj.items()}
            return obj

        return convert_to_numpy(loaded_file)

    return load_for_name


def test_find_secondary_extrema():
    # Valid Cases
    valid_cases = {
        "wavelet_matrix": [
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 10, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array([[0, 0, 0], [0, 0.1, 0], [0, 0.3, 0], [0, 0.2, 0], [0, 0.1, 0]]),
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 3, 0],
                    [0, 1, 0],
                    [0, 4, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ]
            ),
        ],
        "start_idx": [0, 1, 0],
        "end_idx": [7, 5, 6],
        "scale": [1, 1, 1],
        "direction": [1, 1, 1],
        "relevance_threshold": [2, 0.5, 0.1],
        "primary_amplitude": [1.5, 1.5, 1.5],
        "expected_pos": [4, [], 4],
        "expected_val": [10, 0, 4],
    }

    for i in range(len(valid_cases["wavelet_matrix"])):
        extrema_pos, extrema_val = find_secondary_extrema(
            valid_cases["start_idx"][i],
            valid_cases["end_idx"][i],
            valid_cases["wavelet_matrix"][i],
            valid_cases["scale"][i],
            valid_cases["direction"][i],
            valid_cases["relevance_threshold"][i],
            valid_cases["primary_amplitude"][i],
        )
        assert extrema_pos == valid_cases["expected_pos"][i]
        assert extrema_val == valid_cases["expected_val"][i]

    # Invalid Cases
    invalid_cases = {
        "wavelet_matrix": [np.array([]), np.array([1])],
        "start_idx": [4, 5],
        "end_idx": [2, 1, 0],
        "scale": [-1],
        "direction": [0, 2],
        "relevance_threshold": [0, -1],
        "primary_amplitude": [0, -1],
    }

    for param_name, invalid_values in invalid_cases.items():
        for invalid_value in invalid_values:
            with pytest.raises(ValueError):
                find_secondary_extrema(
                    2 if param_name != "start_idx" else invalid_value,
                    3 if param_name != "end_idx" else invalid_value,
                    (
                        np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                        if param_name != "wavelet_matrix"
                        else invalid_value
                    ),
                    1 if param_name != "scale" else invalid_value,
                    1 if param_name != "direction" else invalid_value,
                    (1 if param_name != "relevance_threshold" else invalid_value),
                    1 if param_name != "primary_amplitude" else invalid_value,
                )
    return


def test_refine_simple_p_wave():
    # Valid Cases
    valid_cases = {
        "wavelet_matrix": [
            np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 2, 2], [1, 2, 2], [-1, 2, 2], [0, 2, 2]]),
            np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        ],
        "onset": [0, 0, 0],
        "offset": [3, 3, 3],
        "alternative_scale": [1, 1, 1],
        "current_scale": [0, 0, 0],
        "changes": [[], ["use_current_scale"], []],
        "expected_p_wave_peaks": [-1, -1, None],
        "expected_estimated_onset": [0, 0, None],
        "expected_estimated_offset": [3, 3, None],
    }

    for i in range(len(valid_cases["wavelet_matrix"])):
        p_wave_peaks, estimated_onset, estimated_offset = refine_simple_p_wave(
            valid_cases["onset"][i],
            valid_cases["offset"][i],
            valid_cases["wavelet_matrix"][i],
            valid_cases["alternative_scale"][i],
            valid_cases["current_scale"][i],
            valid_cases["changes"][i],
        )

        assert p_wave_peaks == valid_cases["expected_p_wave_peaks"][i]
        assert estimated_onset == valid_cases["expected_estimated_onset"][i]

    invalid_cases = {
        "onset": [-1],
        "offset": [-1, 1, 0],
        "current_scale": [-1],
        "alternative_scale": [-1],
        "wavelet_matrix": [
            np.array([1]),
            np.array([[], []]),
            np.array([[0]]),
        ],
    }

    for param_name, invalid_values in invalid_cases.items():
        for invalid_value in invalid_values:
            onset, offset = 1, 2
            alternative_scale, current_scale = 1, 0
            wavelet_matrix = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
            with pytest.raises(ValueError):
                refine_simple_p_wave(
                    (invalid_value if param_name == "onset" else onset),
                    (invalid_value if param_name == "offset" else offset),
                    (
                        invalid_value
                        if param_name == "wavelet_matrix"
                        else wavelet_matrix
                    ),
                    (
                        invalid_value
                        if param_name == "alternative_scale"
                        else alternative_scale
                    ),
                    (invalid_value if param_name == "current_scale" else current_scale),
                    [],
                )
    return


def test_refine_complex_p_wave():
    # Valid Cases
    valid_cases = {
        "wavelet_matrix": [
            np.array(
                [
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                ]
            ),
            np.array(
                [
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                ]
            ),
            np.array(
                [
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                ]
            ),
        ],
        "alternative_scale": [1, 2, 2],
        "current_scale": [0, 1, 1],
        "changes": [[], [], ["use_current_scale"]],
        "extrema_positions": [[0, 4, 10], [0, 1, 2], [0, 4, 10]],
        "expected_p_wave_peaks": [-1, None, -1],
        "expected_estimated_onset": [0, None, 0],
        "expected_estimated_offset": [10, None, 10],
        "expected_Pprima": [4, None, 4],
    }

    for i in range(len(valid_cases["wavelet_matrix"])):
        p_wave_peaks, estimated_onset, estimated_offset, Pprima = refine_complex_p_wave(
            valid_cases["extrema_positions"][i],
            valid_cases["wavelet_matrix"][i],
            valid_cases["alternative_scale"][i],
            valid_cases["current_scale"][i],
            valid_cases["changes"][i],
        )
        assert p_wave_peaks == valid_cases["expected_p_wave_peaks"][i]
        assert estimated_onset == valid_cases["expected_estimated_onset"][i]
        assert estimated_offset == valid_cases["expected_estimated_offset"][i]
        assert Pprima == valid_cases["expected_Pprima"][i]

    # Invalid Cases
    invalid_cases = {
        "extrema_positions": [1],
        "current_scale": [-1],
        "alternative_scale": [-1],
        "wavelet_matrix": [
            np.array([1]),
            np.array([[], []]),
            np.array([[0]]),
        ],
    }

    for param_name, invalid_values in invalid_cases.items():
        for invalid_value in invalid_values:
            extrema_positions = [0, 2, 3]
            alternative_scale, current_scale = 1, 0
            wavelet_matrix = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
            with pytest.raises(ValueError):
                refine_complex_p_wave(
                    (
                        invalid_value
                        if param_name == "extrema_positions"
                        else extrema_positions
                    ),
                    (
                        invalid_value
                        if param_name == "wavelet_matrix"
                        else wavelet_matrix
                    ),
                    (
                        invalid_value
                        if param_name == "alternative_scale"
                        else alternative_scale
                    ),
                    (invalid_value if param_name == "current_scale" else current_scale),
                    [],
                )
    return


def test_detect_p_wave_features():
    # Valid Cases
    valid_cases = {
        "wavelet_matrix": [
            np.array(
                [
                    [5, 2, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 10, 5],
                    [5, -10, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 2, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                ]
            ),  # Extended
            np.array(
                [
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                ]
            ),
            np.array(
                [
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                    [5, 1, 5],
                    [5, -1, 5],
                    [5, 2, 5],
                    [5, -2, 5],
                    [5, 1, 5],
                    [5, 0, 5],
                ]
            ),
        ],
        "window_start": [0, 0, 0],
        "window_end": [15, 6, 12],
        "current_scale": [1, 1, 1],
        "qrs_times": [[0, 6, 12], [0, 6, 12], [0, 6, 12]],
        "beat_indices": [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        "i": [1, 1, 1],
        "freq": [500, 500, 500],
        "amplitude_threshold": [0.1, 10.0, 1.0],
        "expected_has_p_wave": [True, False, False],
        "expected_max_amplitude": [2, 0, 0],
        "expected_min_amplitude": [10, 2, 2],
        "expected_max_position": [0, None, None],
        "expected_min_position": [4, 4, 4],
    }

    for i in range(len(valid_cases["wavelet_matrix"])):
        (
            has_p_wave,
            abs_max_amplitude,
            abs_min_amplitude,
            max_position,
            min_position,
        ) = detect_p_wave_features(
            True,
            valid_cases["wavelet_matrix"][i],
            valid_cases["window_start"][i],
            valid_cases["window_end"][i],
            valid_cases["current_scale"][i],
            valid_cases["qrs_times"][i],
            valid_cases["beat_indices"][i],
            valid_cases["i"][i],
            valid_cases["freq"][i],
            valid_cases["amplitude_threshold"][i],
        )
        assert has_p_wave == valid_cases["expected_has_p_wave"][i]
        assert abs_max_amplitude == valid_cases["expected_max_amplitude"][i]
        assert abs_min_amplitude == valid_cases["expected_min_amplitude"][i]
        assert max_position == valid_cases["expected_max_position"][i]
        assert min_position == valid_cases["expected_min_position"][i]

    # Invalid Cases
    invalid_cases = {
        "wavelet_matrix": [
            np.array([1]),
            np.array([[], []]),
            np.array([[0]]),
        ],
        "current_scale": [-1],
        "i": [-1],
        "freq": [-1, 0],
        "amplitude_threshold": [-1.0, 0],
        "beat_indices": [1],
        "qrs_times": [1],
    }

    for param_name, invalid_values in invalid_cases.items():
        for invalid_value in invalid_values:
            # Default valid values
            wavelet_matrix = np.array([[5, 0, 5], [5, 1, 5], [5, -1, 5], [5, 2, 5]])
            window_start = 0
            window_end = 3
            current_scale = 1
            qrs_times = [0, 6, 12]
            beat_indices = [0, 1, 2]
            i_val = 1
            freq = 1
            amplitude_threshold = 1.0

            with pytest.raises(ValueError):
                detect_p_wave_features(
                    True,
                    (
                        invalid_value
                        if param_name == "wavelet_matrix"
                        else wavelet_matrix
                    ),
                    (invalid_value if param_name == "window_start" else window_start),
                    (invalid_value if param_name == "window_end" else window_end),
                    (invalid_value if param_name == "current_scale" else current_scale),
                    (invalid_value if param_name == "qrs_times" else qrs_times),
                    (invalid_value if param_name == "beat_indices" else beat_indices),
                    (invalid_value if param_name == "i" else i_val),
                    (invalid_value if param_name == "freq" else freq),
                    (
                        invalid_value
                        if param_name == "amplitude_threshold"
                        else amplitude_threshold
                    ),
                )
    return


def test_find_p_wave(load_numpy_array, load_dictionary):
    def compare_dicts_with_arrays(dict1, dict2):
        def convert_arrays_to_lists(data):
            if isinstance(data, dict):
                return {
                    key: convert_arrays_to_lists(value) for key, value in data.items()
                }
            elif isinstance(data, list):
                return [convert_arrays_to_lists(item) for item in data]
            elif isinstance(data, np.ndarray):
                return data.tolist()
            else:
                return data

        def compare_and_print(d1, d2, path="root"):
            match = True
            if isinstance(d1, dict) and isinstance(d2, dict):
                for key in set(d1.keys()).union(d2.keys()):
                    if key not in d1:
                        print(f"{path} -> Missing in dict1: {key}")
                        match = False
                    elif key not in d2:
                        print(f"{path} -> Missing in dict2: {key}")
                        match = False
                    else:
                        if not compare_and_print(d1[key], d2[key], f"{path} -> {key}"):
                            match = False
            elif isinstance(d1, list) and isinstance(d2, list):
                if len(d1) != len(d2):
                    print(f"{path} -> List lengths differ: {len(d1)} vs {len(d2)}")
                    match = False
                for i, (item1, item2) in enumerate(zip(d1, d2)):
                    if not compare_and_print(item1, item2, f"{path}[{i}]"):
                        match = False
            elif isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
                if not np.isclose(d1, d2, equal_nan=True):
                    print(f"{path} -> Value mismatch: {d1} vs {d2}")
                    match = False
            else:
                if d1 != d2:
                    print(f"{path} -> Value mismatch: {d1} vs {d2}")
                    match = False
            return match

        dict1_converted = convert_arrays_to_lists(dict1)
        dict2_converted = convert_arrays_to_lists(dict2)
        return compare_and_print(dict1_converted, dict2_converted)

    (
        locations_output_1,
        estimated_onset_array_output_1,
        estimated_offset_array_output_1,
    ) = find_p_wave(
        load_dictionary("p_wave_module_test_signal_info_dict_input_1"),
        load_numpy_array("p_wave_module_test_sample_excerpt_array_input_1"),
        load_numpy_array("p_wave_module_test_qrs_times_array_input_1"),
        load_dictionary("p_wave_module_test_locations_dict_input_1"),
        load_numpy_array("p_wave_module_test_wavelet_matrix_input_1"),
        [0, 9],
        1,
        500,
        load_dictionary("p_wave_module_test_config_params_dict_input_1"),
    )

    # Valid Cases
    # Check Output Form & Type
    assert isinstance(locations_output_1, dict)
    assert isinstance(estimated_onset_array_output_1, list)
    assert isinstance(estimated_offset_array_output_1, list)

    # Compare Algorithm Output to Ground Truth Output
    assert compare_dicts_with_arrays(
        locations_output_1,
        load_dictionary("p_wave_module_test_locations_dict_output_1"),
    )

    for output, expected in [
        (
            estimated_onset_array_output_1,
            load_numpy_array("p_wave_module_test_estimated_onset_array_output_1"),
        ),
        (
            estimated_offset_array_output_1,
            load_numpy_array("p_wave_module_test_estimated_offset_array_output_1"),
        ),
    ]:
        assert np.allclose(output, expected)

    # Invalid Cases
    test_cases = {
        "signal_info": {},
        "sample_excerpt": [1, np.array([1.1, 2]), [], np.array([])],
        "qrs_times": [1, np.array([1.1, 2]), [], np.array([])],
        "locations": {},
        "wavelet_matrix": [1, []],
        "beat_indices": [1, np.array([0.1, 10])],
        "last_annotation": [1.1, np.array([])],
        "fs": {},
        "config_params": {},
    }

    for param_name, invalid_values in test_cases.items():
        for invalid_value in invalid_values:
            with pytest.raises(ValueError):
                find_p_wave(
                    (
                        invalid_value
                        if param_name == "signal_info"
                        else load_dictionary(
                            "p_wave_module_test_signal_info_dict_input_1"
                        )
                    ),
                    (
                        invalid_value
                        if param_name == "sample_excerpt"
                        else load_numpy_array(
                            "p_wave_module_test_sample_excerpt_array_input_1"
                        )
                    ),
                    (
                        invalid_value
                        if param_name == "qrs_times"
                        else load_numpy_array(
                            "p_wave_module_test_qrs_times_array_input_1"
                        )
                    ),
                    (
                        invalid_value
                        if param_name == "locations"
                        else load_dictionary(
                            "p_wave_module_test_locations_dict_input_1"
                        )
                    ),
                    (
                        invalid_value
                        if param_name == "wavelet_matrix"
                        else load_numpy_array(
                            "p_wave_module_test_wavelet_matrix_input_1"
                        )
                    ),
                    (invalid_value if param_name == "beat_indices" else [0, 9]),
                    (invalid_value if param_name == "last_annotation" else 1),
                    ((invalid_value if param_name == "fs" else 500),),
                    (
                        (
                            invalid_value
                            if param_name == "config_params"
                            else load_dictionary(
                                "p_wave_module_test_config_params_dict_input_1"
                            )
                        ),
                    ),
                )

    return


pytest.main(
    [
        "--cov=p_wave_module",
        "--cov-report=term-missing",
        "test_p_wave_module.py",
    ]
)
