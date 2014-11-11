import numpy as np


def ensure_np_array(array_in):
    """
    This function ensures the input is present as a numpy array.

    Parameters
    ----------
    input: list or np array
        The array to be converted.

    Returns
    -------
    output: np.ndarray
        The unchanged input iff it was an np.ndarray, the converted input
        otherwise.
    """
    if not isinstance(array_in, np.ndarray):
        array_in = np.asarray([array_in])
    return array_in