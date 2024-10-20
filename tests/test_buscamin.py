import numpy as np
import pytest

from Subfunction_Module import buscamin


def test_buscamin():
    """
    Test for the "buscamin" function;
    Located in the Subfunction Module of py_pecg.
    """

    # Valid Cases
    Test_Signal_1 = np.zeros(10, dtype=int)
    assert buscamin(Test_Signal_1) == 0
    Test_Signal_2 = np.linspace(1, 10, 10)
    assert (
        buscamin(Test_Signal_2) == 0
    )  # No local minimum, only global minimum
    Test_Signal_3 = np.linspace(10, 1, 10)
    assert (
        buscamin(Test_Signal_3) == 0
    )  # No local minimum, only global minimum
    Test_Signal_4 = (np.linspace(-5, 5, 11)) ** 2
    assert buscamin(Test_Signal_4) == 4  # Parabolic Signal

    # Invalid Cases
    with pytest.raises(ValueError):
        buscamin(np.ones([2, 2]))
    with pytest.raises(ValueError):
        buscamin(np.array([]))


test_buscamin()
