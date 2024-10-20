import numpy as np
import pytest
from Subfunction_Module import searchon
from scipy.signal.windows import triang


def test_searchon():
    """
    Test for the "searchon" function;
    Located in the Subfunction Module of py_pecg.
    """

    #                              ___  /\  ___
    # Modified Triangular Pulse:      \/  \/
    Test_Signal_1 = np.concatenate(
        (
            np.ones(10),
            1 - 0.5 * triang(10, sym=True),
            1 + 1 * triang(10, sym=True),
            1 - 0.5 * triang(10, sym=True),
            np.ones(10),
        )
    )

    # Valid Cases
    assert searchon(np.int64(30), Test_Signal_1, 1) == 20

    # Invalid Cases
    with pytest.raises(TypeError):
        searchon(np.int64(1), np.array([]), 1)
    with pytest.raises(AttributeError):
        searchon(np.int64(1), [], 1)
    with pytest.raises(AttributeError):
        searchon(-1, Test_Signal_1, 1)
    with pytest.raises(ZeroDivisionError):
        searchon(np.int64(30), Test_Signal_1, 0)
    with pytest.raises(ValueError):
        searchon(np.int64(1), np.ones([2, 2]), 1)
    with pytest.raises(ValueError):
        searchon(np.int64(1), np.ones([2, 2, 2]), 1)


test_searchon()
