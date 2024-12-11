import numpy as np
import pytest
from dictionary_module import create_locations_dictionary


def test_create_location_dictionary():
    # Valid Case
    maxlength = 30
    result = create_locations_dictionary(maxlength)
    expected_keys = [
        "P_Wave_Onset",
        "P_Wave_Peak",
        "P_Wave_Offset",
        "P_Wave_Prime",
        "P_Wave_Scale",
        "P_Wave_Type",
        "QRS_Complex_Onset",
        "Q_Wave",
        "R_Peak",
        "R_Prime",
        "S_Wave",
        "QRS_Complex_Offset",
        "qrs",
        "T_Wave_Onset",
        "T_Wave_Peak",
        "T_Wave_Prime",
        "T_Wave_Offset",
        "T_Wave_Type",
        "T_Wave_Scale",
        "QRSpa",
        "QRSpp",
        "QRSmainpos",
        "QRSmaininv",
    ]

    assert list(result.keys()) == expected_keys

    for key in expected_keys:
        assert isinstance(result[key], np.ndarray)
        assert result[key].shape == (maxlength,)
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
