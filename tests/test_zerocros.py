import numpy as np
import pytest

from Subfunction_Module import zerocros


def test_zerocros():
    """
    Test for the "zerocros" function;
    Located in the Subfunction Module of py_pecg.
    """

    # Valid Cases
    Test_Signal_1 = np.zeros(10, dtype=int)
    assert zerocros(Test_Signal_1) == 0
    Test_Signal_2 = np.ones(10, dtype=int)
    assert zerocros(Test_Signal_2).size == 0
    Test_Signal_3 = np.ones(10, dtype=int) * (-1)
    assert zerocros(Test_Signal_3).size == 0
    Test_Signal_4 = np.linspace(9, -9, 19)
    assert zerocros(Test_Signal_4) == 9

    # Invalid Cases
    with pytest.raises(ValueError):
        zerocros(np.ones([2, 2]))
    with pytest.raises(ValueError):
        zerocros(np.ones([2, 2, 2]))
    with pytest.raises(ValueError):
        zerocros(np.array([]))


test_zerocros()
