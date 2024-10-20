import numpy as np
import pytest

from Subfunction_Module import modmax


def test_modmax():
    """
    Test for the "modmax" function;
    Located in the Subfunction Module of py_pecg.
    """

    # Valid Cases
    # Zero Array
    Test_Signal_1 = np.zeros(10, dtype=int)
    assert modmax(Test_Signal_1, 0, 0, 1).size == 0
    assert modmax(Test_Signal_1, 0, 0, -1).size == 0
    assert modmax(Test_Signal_1, 0, 0, 0).size == 0

    # Sine Wave
    Test_Signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert modmax(Test_Signal_2, 0, 2, 1).size == 0
    assert modmax(Test_Signal_2, 0, 2, -1).size == 0
    assert modmax(Test_Signal_2, 0, 2, 0).size == 0
    assert modmax(Test_Signal_2, len(Test_Signal_2), 2, 0).size == 0
    assert np.array_equal(
        modmax(Test_Signal_2, 0, 0, 1),
        np.array(
            [
                [5],
                [25],
                [45],
                [65],
                [85],
                [104],
                [124],
                [144],
                [164],
                [184],
            ]
        ),
    )

    assert np.array_equal(
        modmax(Test_Signal_2, 0, 0, -1),
        np.array(
            [
                [15],
                [35],
                [55],
                [75],
                [95],
                [114],
                [134],
                [154],
                [174],
                [194],
            ]
        ),
    )

    assert np.array_equal(
        modmax(Test_Signal_2, 0, 0, 0),
        np.array(
            [
                [5],
                [15],
                [25],
                [35],
                [45],
                [55],
                [65],
                [75],
                [85],
                [95],
                [104],
                [114],
                [124],
                [134],
                [144],
                [154],
                [164],
                [174],
                [184],
                [194],
            ]
        ),
    )

    # Invalid Cases
    with pytest.raises(ValueError):
        modmax(np.ones([2, 2]), 0, 0, 0)
    with pytest.raises(ValueError):
        modmax(np.ones([2, 2, 2]), 0, 0, 0)
    with pytest.raises(ValueError):
        modmax(np.array([]), 0, 0, 0)
    with pytest.raises(ValueError):
        modmax(Test_Signal_1, -1, 0, 0)


test_modmax()
