import numpy as np
import pytest
from scipy.signal.windows import triang
from utility_module import (
    find_first_peak,
    find_first_zero_crossing,
    find_last_peak,
    find_local_modulus_minimum,
    find_modulus_extrema,
    search_offset,
    search_onset,
)


def test_find_local_modulus_minimum():
    # Valid Cases
    assert find_local_modulus_minimum(np.zeros(10, dtype=int)) == 0
    assert (
        find_local_modulus_minimum(np.linspace(1, 10, 10)) == 0
    )  # No local minimum, only global minimum
    assert (
        find_local_modulus_minimum(np.linspace(10, 1, 10)) == 0
    )  # No local minimum, only global minimum
    assert (
        find_local_modulus_minimum((np.linspace(-5, 5, 11)) ** 2) == 4
    )  # Parabolic Signal

    # Invalid Cases
    with pytest.raises(ValueError):
        find_local_modulus_minimum(np.ones([2, 2]))
    with pytest.raises(ValueError):
        find_local_modulus_minimum(np.array([]))
    with pytest.raises(ValueError):
        find_local_modulus_minimum(np.zeros((2, 3, 4)))
    return


@pytest.fixture
def modified_triangular_pulse():
    #                              ___  /\  ___
    # Modified Triangular Pulse:      \/  \/
    modified_triangular_pulse = np.concatenate(
        (
            np.ones(10),
            1 - 0.5 * triang(10, sym=True),
            1 + 1 * triang(10, sym=True),
            1 - 0.5 * triang(10, sym=True),
            np.ones(10),
        )
    )
    return modified_triangular_pulse


@pytest.fixture
def setup_for_onset_and_offset(modified_triangular_pulse):
    def check_invalid_cases(function):
        with pytest.raises(ValueError):
            function(np.int64(1), np.array([]), 1)
        with pytest.raises(ValueError):
            function(np.int64(1), [], 1)
        with pytest.raises(AttributeError):
            function(-1, modified_triangular_pulse, 1)
        with pytest.raises(ZeroDivisionError):
            function(np.int64(30), modified_triangular_pulse, 0)
        with pytest.raises(ValueError):
            function(np.int64(1), np.ones([2, 2]), 1)
        with pytest.raises(ValueError):
            function(np.int64(1), np.ones([2, 2, 2]), 1)

    check_invalid_cases(search_onset)
    check_invalid_cases(search_offset)


def test_search_onset(modified_triangular_pulse, setup_for_onset_and_offset):
    # Sample Signal with Local Minima
    test_signal_2 = np.array([1, 0.5, 0.2, 0.3, 0.6, 1, 0.8, 0.4, 0.1])

    # Valid Cases:
    assert search_onset(np.int64(30), modified_triangular_pulse, 1) == 20
    assert search_onset(np.int64(0), np.zeros(10), 1) == 0  # Flat Signal
    assert search_onset(np.int64(0), np.ones(10), 1) == 0  # Flat Signal
    assert search_onset(np.int64(30), test_signal_2, 1) == 24
    return


def test_search_offset(modified_triangular_pulse, setup_for_onset_and_offset):
    # Valid Cases
    assert search_offset(np.int64(30), modified_triangular_pulse, 1) == 39
    assert search_offset(np.int64(0), np.ones(10), 0.1) == 9  # Flat Signal
    return


def test_find_last_peak():
    # Valid Cases
    # Zero Array
    test_signal_1 = np.zeros(10, dtype=int)
    assert find_last_peak(test_signal_1, 0) == -1
    # Sine Wave
    test_signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert find_last_peak(test_signal_2, 0) == -5

    # Invalid Cases
    with pytest.raises(ValueError):
        find_last_peak(np.ones([2, 2]), 0)
    with pytest.raises(ValueError):
        find_last_peak(np.ones([2, 2, 2]), 0)
    with pytest.raises(ValueError):
        find_last_peak(np.array([]), 0)
    with pytest.raises(ValueError):
        find_last_peak(test_signal_1, -1)
    return


def test_find_first_peak():
    # Valid Cases
    # Zero Array
    test_signal_1 = np.zeros(10, dtype=int)
    assert find_first_peak(test_signal_1, 0) == 1
    # Sine Wave
    test_signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert find_first_peak(test_signal_2, 0) == 5

    # Invalid Cases
    with pytest.raises(ValueError):
        find_first_peak(np.ones([2, 2]), 0)
    with pytest.raises(ValueError):
        find_first_peak(np.ones([2, 2, 2]), 0)
    with pytest.raises(ValueError):
        find_first_peak(np.array([]), 0)
    with pytest.raises(ValueError):
        find_first_peak(test_signal_1, -1)
    return


def test_find_first_zero_crossing():
    # Valid Cases
    test_signal_1 = np.zeros(10, dtype=int)
    assert find_first_zero_crossing(test_signal_1) == 0
    test_signal_2 = np.ones(10, dtype=int)
    assert find_first_zero_crossing(test_signal_2).size == 0
    test_signal_3 = np.ones(10, dtype=int) * (-1)
    assert find_first_zero_crossing(test_signal_3).size == 0
    test_signal_4 = np.linspace(9, -9, 19)
    assert find_first_zero_crossing(test_signal_4) == 9

    # Invalid Cases
    with pytest.raises(ValueError):
        find_first_zero_crossing(np.ones([2, 2]))
    with pytest.raises(ValueError):
        find_first_zero_crossing(np.ones([2, 2, 2]))
    with pytest.raises(ValueError):
        find_first_zero_crossing(np.array([]))
    return


def test_find_modulus_extrema():
    # Valid Cases
    # Zero Array
    assert find_modulus_extrema(np.zeros(10, dtype=int), 0, 0, 1).size == 0
    assert find_modulus_extrema(np.zeros(10, dtype=int), 0, 0, -1).size == 0
    assert find_modulus_extrema(np.zeros(10, dtype=int), 0, 0, 0).size == 0

    # Sine Wave
    test_signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert find_modulus_extrema(test_signal_2, 0, 2, 1).size == 0
    assert find_modulus_extrema(test_signal_2, 0, 2, -1).size == 0
    assert find_modulus_extrema(test_signal_2, 0, 2, 0).size == 0
    assert find_modulus_extrema(test_signal_2, len(test_signal_2), 2, 0).size == 0
    assert np.array_equal(
        find_modulus_extrema(test_signal_2, 0, 0, 1),
        np.array(
            [
                5,
                25,
                45,
                65,
                85,
                104,
                124,
                144,
                164,
                184,
            ]
        ),
    )

    assert np.array_equal(
        find_modulus_extrema(test_signal_2, 0, 0, -1),
        np.array(
            [
                15,
                35,
                55,
                75,
                95,
                114,
                134,
                154,
                174,
                194,
            ]
        ),
    )

    assert np.array_equal(
        find_modulus_extrema(test_signal_2, 0, 0, 0),
        np.array(
            [
                5,
                15,
                25,
                35,
                45,
                55,
                65,
                75,
                85,
                95,
                104,
                114,
                124,
                134,
                144,
                154,
                164,
                174,
                184,
                194,
            ]
        ),
    )

    # Invalid Cases
    with pytest.raises(ValueError):
        find_modulus_extrema(np.ones([2, 2]), 0, 0, 0)
    with pytest.raises(ValueError):
        find_modulus_extrema(np.ones([2, 2, 2]), 0, 0, 0)
    with pytest.raises(ValueError):
        find_modulus_extrema(np.array([]), 0, 0, 0)
    with pytest.raises(ValueError):
        find_modulus_extrema(np.zeros(10, dtype=int), -1, 0, 0)
    return


pytest.main(
    [
        "--cov=utility_module",
        "--cov-report=term-missing",
        "--disable-warnings",
        "test_utility_module.py",
    ]
)
