__author__ = 'Frederik Diehl'

import numpy as np


class Candidate:

    params = None
    params_used = None
    result = None
    cost = None
    valid = None
    worker_information = None

    def __init__(self, params, params_used=None, result=None, valid=True, worker_information=None):
        """
        Initializes the candidate object. It requires only the used parameter vector, the rest are optional information.
        :param params: A numpy vector (of floats) of parameter values representing this candidate's parameters used.
        :param params_used: An (optional) vector of boolean values specifying whether a certain parameter from params
        is used. Use this function for example when describing a neural network where information about the third
        hidden layer is irrelevant when having less than three layers used.
        :param result: The currently achieved result for this candidate. May be None in the beginning.
        :param valid: Whether the candidate's result is valid.
        This may be False if, for example, the function to optimize crashes at certain (unknown) values.
        :param worker_information: A dictionary representing information for the workers. Keys should be strings.
        Can be used to, for example, store file paths where information necessary for continuation is stored.
        """
        #TODO add parameter validity check.
        self.params = params

        if params_used is None:
            params_used = np.ones(params.shape, dtype=bool)
        self.params_used = params_used

        self.result = result
        self.cost = 0
        self.valid = valid

        if worker_information is None:
            worker_information = {}
        self.worker_information = worker_information