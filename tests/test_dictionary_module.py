import numpy as np
import pytest

from dictionary_module import (
    create_locations_dictionary,
)


def test_create_location_dictionary():
    # Valid Case
    maxlength = 30
    result = create_locations_dictionary(maxlength)
    expected_keys = [
        "Pon",
        "P",
        "Poff",
        "Pprima",
        "Pscale",
        "Ptipo",
        "QRSon",
        "Q",
        "R",
        "Rprima",
        "S",
        "QRSoff",
        "qrs",
        "Ton",
        "T",
        "Tprima",
        "Toff",
        "Ttipo",
        "Tscale",
        "QRSpa",
        "QRSpp",
        "QRSmainpos",
        "QRSmaininv",
    ]

    assert list(result.keys()) == expected_keys

    for key in expected_keys:
        assert isinstance(result[key], np.ndarray)
        assert result[key].shape == (1, maxlength)
        assert np.all(np.isnan(result[key]))

    # Invalid Cases
    with pytest.raises(ValueError):
        create_locations_dictionary(-1)
    with pytest.raises(ValueError):
        create_locations_dictionary(0)
    with pytest.raises(ValueError):
        create_locations_dictionary(np.array([1, 1]))

pytest.main(
    [
        "--cov=dictionary_module",
        "--cov-report=term-missing",
        "--disable-warnings",
        "test_dictionary_module.py",
    ]
)
