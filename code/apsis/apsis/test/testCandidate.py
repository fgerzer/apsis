from apsis.models import Candidate

__author__ = 'Frederik Diehl'

import numpy as np
import random


# noinspection PyPep8Naming,PyPep8Naming
class testCandidate(object):

    test_point = None
    test_candidate = None

    def setUp(self):
        self.test_point = [1, 1, 1]
        for i in range(0, len(self.test_point)):
            self.test_point[i] = random.gauss(0, 10)
        self.test_candidate = Candidate(self.test_point)

    def test_initialization(self):
        assert self.test_candidate is not None
        assert isinstance(self.test_candidate, Candidate)
        assert (self.test_candidate.params == self.test_point).all()