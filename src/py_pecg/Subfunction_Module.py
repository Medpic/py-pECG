import numpy as np  # pragma: no cover


def buscamin(
    x: np.ndarray,
) -> int:
    """
    This function searches the first local minimum of the modulus of an array x,
    While truncating the edges (First and Final Indices).

    Args:
        x (np.ndarray): The target array.

    Returns:
        ind (int): The first index in x for which the first minimum is located.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if x.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif x.ndim == 2 and (x.shape[0] > 1 and x.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")

    x = np.abs(x)
    localmin = (x[1:-1] <= x[:-2]) & (x[1:-1] <= x[2:])
    ind = np.argmax(localmin)
    return ind


def searchon(piconset: int, sig: np.ndarray, K: float) -> int:
    """
    This function searches the onset of a wave using the derivative method.

    Args:
        piconset (int): Position of the first relevant onset in the wavelet.
        sig (np.ndarray): Wavelet Signal (in a single scale).
        K (float): Threshold Factor.

    Returns:
        onset (int): Resultant index of the wave onset.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if K == 0:
        raise ZeroDivisionError("K cannot be zero!")

    if sig.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif sig.ndim == 2 and (sig.shape[0] > 1 and sig.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")

    if piconset.size == 0 or sig.size == 0:
        onset = []

    else:
        maxderon = abs(sig[-1])

        # Maximum Derivative
        ind1 = (
            min(np.argwhere(abs(np.flipud(sig[0:-1])) < maxderon / K))
            if np.any(abs(np.flipud(sig[0:-2])) < maxderon / K)
            else np.array([])
        )
        if ind1.size != 0:
            ind1 = ind1[0]
        ind2 = buscamin(np.flipud(sig[0:]))

        if (ind1.size == 0 or ind1 == 0) and (ind2.size == 0 or ind2 == 0):
            onset = piconset - len(sig) + 1
        elif ind1.size == 0 or ind1 == 0:
            onset = piconset - ind2 - 1
        elif ind2.size == 0 or ind2 == 0:
            onset = piconset - ind1 - 1
        else:
            onset = piconset - min(ind1, ind2) - 1
    return max(onset, 0)


def searchoff(picoffset: int, sig: np.ndarray, K: float) -> int:
    """
    This function searches the offset of a wave using the derivative method.

    Args:
        picoffset (int): Position of the first relevant offset in the wavelet.
        sig (np.ndarray): Wavelet Signal (in a single scale).
        K (float): Threshold Factor.

    Returns:
        offset (int): Resultant index of the wave offset.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if K == 0:
        raise ZeroDivisionError("K cannot be zero!")

    if sig.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif sig.ndim == 2 and (sig.shape[0] > 1 and sig.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")

    if picoffset.size == 0 or sig.size == 0:
        offset = []

    else:

        maxderoff = abs(sig[0])

        # maximum derivative
        ind1 = (
            min(np.argwhere(abs(sig[1:]) < maxderoff / K))
            if np.any((abs(sig[1:])) < maxderoff / K)
            else np.array([])
        )
        ind1 = ind1[0]
        ind2 = buscamin(sig[1:])

        if (ind1.size == 0 or ind1 == 0) and (ind2.size == 0 or ind2 == 0):
            offset = picoffset + len(sig) - 1
        elif ind1.size == 0 or ind1 == 0:
            offset = picoffset + ind2
        elif ind2.size == 0 or ind2 == 0:
            offset = picoffset + ind1
        else:
            offset = picoffset + min(ind1, ind2) + 1
    return max(offset, 0)


def picant(sig: np.ndarray, time: int) -> int:
    """
    This function calculates the first peak of a signal, nearest to the end.

    Args:
        sig (np.ndarray): The signal array.
        time (int): Time relevant to the scale of the target signal.

    Returns:
        pa (int): Index of the maximum in the target signal.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if sig.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif sig.ndim == 2 and (sig.shape[0] > 1 and sig.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")
    elif sig.size == 0:
        raise ValueError("Input an array which is not empty.")
    if time < 0:
        raise ValueError("Input a non-negative time.")

    sig = np.flipud(sig)
    der = np.diff(sig)
    cero = min(np.argwhere((der[0:-1] * der[1:] <= 0)))
    pa = time - cero - 1
    return pa


def picpost(sig: np.ndarray, time: int) -> int:
    """
    This function calculates the first peak of a signal, nearest to the beginning.

    Args:
        sig (np.ndarray): The signal array.
        time (int): Time relevant to the scale of the target signal.

    Returns:
        pp (int): Index of the maximum in the target signal.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if sig.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif sig.ndim == 2 and (sig.shape[0] > 1 and sig.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")
    elif sig.size == 0:
        raise ValueError("Input an array which is not empty.")
    if time < 0:
        raise ValueError("Input a non-negative time.")
    der = np.diff(sig)
    cero = min(np.argwhere((der[0:-2] * der[1:-1] <= 0)))
    pp = time + cero + 1
    return pp


def zerocros(x: np.ndarray) -> int:
    """
    This function returns the index of the input vector in which the first
    zero crossing is located.

    Args:
        x (np.ndarray): The target array.

    Returns:
        index (int): The first index in which zero crossing occurs.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if x.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif x.ndim == 2 and (x.shape[0] > 1 and x.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")
    elif x.size == 0:
        raise ValueError("Input an array which is not empty.")

    m = x[1:-1] * x[0:-2]
    if np.argwhere(m <= 0).size == 0:
        return np.array([])
    index = min(np.argwhere(m <= 0))
    if abs(x[index]) > abs(x[index + 1]):
        index = index + 1
    return index


def modmax(
    x: np.ndarray, first_samp: int, threshold: float, signo: float
) -> np.ndarray:
    """
    Returns the indices of local modulus maxima or minima in vector x
    where the modulus exceeds the specified threshold.

    Args:
        x (np.ndarray): The target signal array.
        first_samp (int): Analyze signal from first_samp sample.
        threshold (float): An amplitude threshold to consider maxima.
        signo (float): Sign of the maxima.
                       If the specified objective is 0, searches for
                       both positive and negative maxima.
                       If the specified objective is +1 or -1, searches
                       for modulus maxima positive or negative.

    Returns:
        indexes (np.ndarray): The indices of the modulus maxima.

    Credits:
        Original authors: Juan Pablo Martínez, Rute Almeida, Salvador Olmos,
        Ana Paula Rocha, Pablo Laguna
        Original Publication: A wavelet-based ECG delineator:
        evaluation on standard databases
        Original DOI: 10.1109/TBME.2003.821031.
    """
    if x.ndim > 2:
        raise ValueError("Input is a higher-dimensional array, not a vector.")

    elif x.ndim == 2 and (x.shape[0] > 1 and x.shape[1] > 1):
        raise ValueError("Input is a matrix, not a row or column vector.")
    elif x.size == 0:
        raise ValueError("Input an array which is not empty.")
    if first_samp < 0:
        raise ValueError("Input a non-negative first_samp.")

    lx = len(x) - 1
    indexes = []
    first_samp = int(first_samp)

    if np.size(first_samp) < 2:
        first_samp = [max(1, first_samp), lx]

    if lx > first_samp[0]:
        s = np.sign(x)
        x = abs(x)

        sample_curr_idx = np.arange(first_samp[0], first_samp[1])
        sample_prev_idx = np.arange(first_samp[0] - 1, first_samp[1] - 1)
        sample_next_idx = np.arange(first_samp[0] + 1, first_samp[1] + 1)

        # FLOAT POINT PERCISION
        localmax = (
            (x[sample_curr_idx] >= x[sample_prev_idx])
            & (x[sample_curr_idx] > x[sample_next_idx])
            & (x[sample_curr_idx] >= threshold)
            & (s[sample_curr_idx] * signo >= 0)
        )  # if 0, it doesn't matter

        iAux = np.zeros_like(x, dtype=bool)
        iAux[sample_curr_idx] = localmax
        indexes = np.argwhere(iAux)

    else:
        return np.array([])

    return indexes
