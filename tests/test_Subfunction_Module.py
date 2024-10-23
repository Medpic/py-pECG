import numpy as np  # pragma: no cover
import pytest  # pragma: no cover
from scipy.signal.windows import triang  # pragma: no cover

from Subfunction_Module import (
    buscamin,
    modmax,
    picant,
    picpost,
    searchoff,
    searchon,
    zerocros,
)  # pragma: no cover


@pytest.mark.buscamin_testmarker
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
    with pytest.raises(ValueError):
        buscamin(np.zeros((2, 3, 4)))
    return


@pytest.mark.searchon_testmarker
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

    # Flat Signals
    Test_Signal_2 = np.zeros(10)
    Test_Signal_3 = np.ones(10)
    # Sample Signal with Local Minima
    Test_Signal_4 = np.array([1, 0.5, 0.2, 0.3, 0.6, 1, 0.8, 0.4, 0.1])

    # Valid Cases:
    assert searchon(np.int64(30), Test_Signal_1, 1) == 20
    assert searchon(np.int64(0), Test_Signal_2, 1) == 0
    assert searchon(np.int64(0), Test_Signal_3, 1) == 0
    assert searchon(np.int64(30), Test_Signal_4, 1) == 24

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
    return


@pytest.mark.searchoff_testmarker
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
    # Flat Signals
    Test_Signal_3 = np.ones(10)

    # Valid Cases
    assert searchoff(np.int64(30), Test_Signal_1, 1) == 39
    assert searchoff(np.int64(0), Test_Signal_3, 0.1) == 9

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
    return


@pytest.mark.picant_testmarker
def test_picant():
    """
    Test for the "picant" function;
    Located in the Subfunction Module of py_pecg.
    """

    # Valid Cases
    # Zero Array
    Test_Signal_1 = np.zeros(10, dtype=int)
    assert picant(Test_Signal_1, 0) == -1
    # Sine Wave
    Test_Signal_2 = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 2 * 100))
    assert picant(Test_Signal_2, 0) == -5

    # Invalid Cases
    with pytest.raises(ValueError):
        picant(np.ones([2, 2]), 0)
    with pytest.raises(ValueError):
        picant(np.ones([2, 2, 2]), 0)
    with pytest.raises(ValueError):
        picant(np.array([]), 0)
    with pytest.raises(ValueError):
        picant(Test_Signal_1, -1)
    return


@pytest.mark.picpost_testmarker
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
    return


@pytest.mark.zerocros_testmarker
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
    return


@pytest.mark.modmax_testmarker
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
    return


if __name__ == "__main__":
    pytest.main(
        [
            "--cov=Subfunction_Module",
            "--cov-report=term-missing",
            "--disable-warnings",
            "test_Subfunction_Module.py",
        ]
    )
