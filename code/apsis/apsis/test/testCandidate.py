__author__ = 'Frederik Diehl'

from apsis.Candidate import Candidate
import numpy as np
import random


# noinspection PyPep8Naming,PyPep8Naming
class testCandidate():

    test_point = None
    test_candidate = None

    def setUp(self):
        self.test_point = np.ones((3, 1))
        for i in range(0, self.test_point.shape[0]):
            self.test_point[i] = random.gauss(0, 10)
        self.test_candidate = Candidate(self.test_point)

    def test_initialization(self):
        assert self.test_candidate is not None
        assert isinstance(self.test_candidate, Candidate)
        assert (self.test_candidate.params == self.test_point).all