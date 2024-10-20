import numpy as np
import pytest

from Subfunction_Module import picpost


def test_picpost():
    """
    Test for the "picpost" function;
    Located in the Subfunction Module of py_pecg.
    """

    # Valid Cases
    # Zero Array
    Test_Signal_1 = np.zeros(10, dtype=int)
    assert picpost(Test_Signal_1, 0) == 1
    # Sine Wave
    Test_Signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert picpost(Test_Signal_2, 0) == 5

    # Invalid Cases
    with pytest.raises(ValueError):
        picpost(np.ones([2, 2]), 0)
    with pytest.raises(ValueError):
        picpost(np.ones([2, 2, 2]), 0)
    with pytest.raises(ValueError):
        picpost(np.array([]), 0)
    with pytest.raises(ValueError):
        picpost(Test_Signal_1, -1)


test_picpost()
