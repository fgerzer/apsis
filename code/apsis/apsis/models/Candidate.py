__author__ = 'Frederik Diehl'

import numpy as np


class Candidate(object):
    """
    Represents a candidate, that is in a mathematical sense a single vector of
    parameter values that is used in optimization.

    Detailed Information
    --------------------
    This class is used for optimizer-worker communications, and intra-optimizer
    storage. It stores the parameter vector, the result that has been achieved
    on the problem when using this particular parameter vecotr,
    cost and validity of the value and an optional field for identifying
    worker-relevant data.
    """

    params = None
    params_used = None
    result = None
    cost = None
    valid = None
    worker_information = None

    def __init__(self, params, params_used=None, result=None, valid=True,
                 worker_information=None):
        """
        Initializes the candidate object.

        It requires only the used parameter vector, the rest are optional
        information.

        Parameters
        ----------
        params: vector
            A list of parameter values representing this candidate's parameters
            used.
        params_used: vector of booleans
            An (optional) vector of boolean values specifying whether a certain
            parameter from params is used.
            Use this function for example when describing a neural network
            where information about the third hidden layer is irrelevant when
            having less than three layers used.
        result: float
            The currently achieved result for this candidate. May be None in
            the beginning.
        valid: bool
            Whether the candidate's result is valid. This may be False if, for
            example, the function to optimize crashes at certain (unknown)
            values.
        worker_information: dict of string keys
            A dictionary representing information for the workers.
            Keys should be strings. Can be used to, for example, store file
            paths where information necessary for continuation is stored.

        Raises
        ------
        ValueError:
            If no params vector is given.

        """
        if params is None:
           raise ValueError("No param vector given!")

        self.params = params

        if params_used is None:
            params_used = [True]*len(self.params)
        self.params_used = params_used

        self.result = result
        self.cost = 0
        self.valid = valid

        if worker_information is None:
            worker_information = {}
        self.worker_information = worker_information

    def __eq__(self, other):
        """
        Compares two Candidates.

        Candidates are defined as being equal iff their params vectors are
        equal.

        Parameters
        ----------
        other: Candidate
            The other Candidate to be compared to.

        Returns
        -------
            equal: bool
                True iff the Candidates are equal. False otherwise.
        """
        if not isinstance(other, Candidate):
            return False
        if self.params == other.params:
            return True
        return False

    def __hash__(self):
        """
        Hash method for Candidate. To be consistant with this class' __eq__.

        The list is first converted to string, then hashed. Since lists
        are not hashable in python.

        Returns
        ----------
        hash_value: String
            Hashed string of the string representing this Candidate vecotr of
            hyper params.

        :return:
        """
        #TODO this is a bit odd at the moemnt, since there is a good reason for
        #lists not beeing hashable. Hence to make this better we should have
        #a element by element __eq__ method, then this __hash__ makes more
        #  sense
        hash_value = hash(str(self.params))

        return hash_value


    def __str__(self):
        """
        Stringifies the object. Currently only prints the parameters and the
        result value.

        Returns
        ----------
        string: stringified object
        """
        string = "\n"
        string += "params: " + str(self.params) + "\n"
        string += "result: " + str(self.result) + "\n"
        return string

    def __lt__(self, other):
        """
        Compares two Candidate instances. Compares their result fields.

        Parameters
        ----------
        other: Candidate
            A candidate that shall be used for comparison with this object.

        Returns
        ----------
            lt: bool
                True if the result of this candidate is smaller than
                the other candidate's result.

        Raises
        ------
        ValueError:
            If other is not a Candidate object.
        """
        #TODO Candidate lt is not consistent with candidate equals! Comparison
        #of results with < is not such a good idea!
        if not isinstance(other, Candidate):
            raise ValueError("Is not compared with a Candidate, but with "
                             + str(other) + ".")
        if self.result < other.result:
            return True
        return False

    def as_vector(self):
        """
        Convert this Candidate's parameter vector to a numpy array.

        Returns
        ----------
            params: numpy.array
                The parameter vector represented by this Candidate object
                as numpy array.

        """
        return np.array(self.params)