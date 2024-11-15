from numpy import isscalar, nan, zeros


def create_locations_dictionary(maxlength: int) -> dict:
    """
    Creates the "Location" dictionary, used for extracting ECG Parameters.

    Args:
        maxlength (int): Estimated maximum number of beats
        in the ECG signal excerpt.

    Returns:
        locations (dict): Dictionary containing information relevant to the P, QRS and T intervals
        in the ECG signal excerpt.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos, Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator: evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.

    """

    if not isscalar(maxlength):
        raise ValueError(
            "Input must be a scalar, not an array or higher-dimensional object."
        )

    if maxlength <= 0:
        raise ValueError("maxlength must be a positive integer.")

    nanvec = zeros((1, maxlength))
    nanvec[:] = nan

    locations = {
        "Pon": nanvec.copy(),
        "P": nanvec.copy(),
        "Poff": nanvec.copy(),
        "Pprima": nanvec.copy(),
        "Pscale": nanvec.copy(),
        "Ptipo": nanvec.copy(),
        "QRSon": nanvec.copy(),
        "Q": nanvec.copy(),
        "R": nanvec.copy(),
        "Rprima": nanvec.copy(),
        "S": nanvec.copy(),
        "QRSoff": nanvec.copy(),
        "qrs": nanvec.copy(),
        "Ton": nanvec.copy(),
        "T": nanvec.copy(),
        "Tprima": nanvec.copy(),
        "Toff": nanvec.copy(),
        "Ttipo": nanvec.copy(),
        "Tscale": nanvec.copy(),
        "QRSpa": nanvec.copy(),
        "QRSpp": nanvec.copy(),
        "QRSmainpos": nanvec.copy(),
        "QRSmaininv": nanvec.copy(),
    }

    return locations
