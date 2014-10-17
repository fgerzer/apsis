from apsis.models.Candidate import Candidate
import nose.tools as nt

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
        assert self.test_candidate.params == self.test_point


    def test_equals_hash_consistency(self):
        other_test_point_same_point = self.test_point = [self.test_point[0], self.test_point[1], self.test_point[2]]
        other_test_candidate_same_point = Candidate(other_test_point_same_point)

        assert self.test_candidate.__hash__() == other_test_candidate_same_point.__hash__()