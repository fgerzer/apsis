__author__ = 'Frederik Diehl'
import numpy as np


def check_array(array):
    """
    Checks the array validity and convert it to np.array.

    :param array: The array to check. Has to be at least 2d, and a numpy array.
    :return: The converted array.
    """
    array = np.atleast_2d(array)
    array = np.array(array)
    return array


def check_array_dimensions_equal(array1, array2):
    """
    Checks whether two arrays are equal and valid.

    :param array1:
    :param array2:
    :return:
    """
    array1 = check_array(array1)
    array2 = check_array(array2)
    return array1.shape == array2.shape