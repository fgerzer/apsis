import numpy as np


def check_array(array):
    """
    Checks the array validity and convert it to np.array.

    Parameters
    ----------
    array: list or np.array
        The array to check. Has to be at least 2d, and a numpy array.

    Returns
    -------
    array: np.array (2-dimensional)
        The converted array
    """
    array = np.atleast_2d(array)
    array = np.array(array)
    return array


def check_array_dimensions_equal(array1, array2):
    """
    Checks whether two arrays are valid and their dimensions equal.

    Parameters
    ----------
    array1: list or np.array
        The first array to be checked

    array2: list or np.array
        The second array to be checked

    Returns
    -------
    is_equal: bool
        True iff the dimensions of both arrays are equal.
    """
    array1 = check_array(array1)
    array2 = check_array(array2)
    return array1.shape == array2.shape