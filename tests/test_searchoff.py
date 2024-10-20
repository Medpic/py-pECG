import numpy as np
import pytest
from Subfunction_Module import searchoff
from scipy.signal.windows import triang


def test_searchoff():
    """
    Test for the "test_searchoff" function;
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
    assert searchoff(np.int64(30), Test_Signal_1, 1) == 39

    # Invalid Cases
    with pytest.raises(TypeError):
        searchoff(np.int64(1), np.array([]), 1)
    with pytest.raises(AttributeError):
        searchoff(np.int64(1), [], 1)
    with pytest.raises(AttributeError):
        searchoff(-1, Test_Signal_1, 1)
    with pytest.raises(ZeroDivisionError):
        searchoff(np.int64(30), Test_Signal_1, 0)
    with pytest.raises(ValueError):
        searchoff(np.int64(1), np.ones([2, 2]), 1)
    with pytest.raises(ValueError):
        searchoff(np.int64(1), np.ones([2, 2, 2]), 1)


test_searchoff()
