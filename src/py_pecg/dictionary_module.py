from numpy import nan, squeeze, zeros


def create_locations_dictionary(maxlength: int) -> dict:
    """
    Creates the "Location" dictionary, used for extracting ECG Parameters.

    Args:
        maxlength (int): Estimated maximum number of beats in the ECG
            signal excerpt.

    Returns:
        locations (dict): Dictionary containing information relevant to the
            P, QRS, and T intervals in the ECG signal excerpt.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
            Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator: evaluation on
            standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """

    if not isinstance(maxlength, int):
        raise ValueError(
            "Input must be a scalar, not an array or higher-dimensional object."
        )

    if maxlength <= 0:
        raise ValueError("maxlength must be a non-negative integer.")

    nanvec = squeeze(zeros((1, maxlength)))
    nanvec[:] = nan

    locations = {
        "P_Wave_Onset": nanvec.copy(),
        "P_Wave_Peak": nanvec.copy(),
        "P_Wave_Offset": nanvec.copy(),
        "P_Wave_Prime": nanvec.copy(),
        "P_Wave_Scale": nanvec.copy(),
        "P_Wave_Type": nanvec.copy(),
        "QRS_Complex_Onset": nanvec.copy(),
        "Q_Wave": nanvec.copy(),
        "R_Peak": nanvec.copy(),
        "R_Prime": nanvec.copy(),
        "S_Wave": nanvec.copy(),
        "QRS_Complex_Offset": nanvec.copy(),
        "qrs": nanvec.copy(),
        "T_Wave_Onset": nanvec.copy(),
        "T_Wave_Peak": nanvec.copy(),
        "T_Wave_Prime": nanvec.copy(),
        "T_Wave_Offset": nanvec.copy(),
        "T_Wave_Type": nanvec.copy(),
        "T_Wave_Scale": nanvec.copy(),
        "QRSpa": nanvec.copy(),
        "QRSpp": nanvec.copy(),
        "QRSmainpos": nanvec.copy(),
        "QRSmaininv": nanvec.copy(),
    }

    return locations
