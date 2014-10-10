#!/usr/bin/python

from apsis.OptimizationCoreInterface import OptimizationCoreInterface
from sklearn.utils import check_random_state
import numpy as np


class RandomSearchCore(OptimizationCoreInterface):
    lower_bound = None
    upper_bound = None
    random_state = None

    def __init__(self, lower_bound, upper_bound, random_state=0):
        print("Initializing Random Search Core for bounds..." + str(lower_bound) + " and " + str(upper_bound))

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random_state = check_random_state(random_state)

    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        print("Worker sending worker information")

    def next_candidate(self, worker_id=None):
        new_candidate = np.zeros(self.lower_bound.shape)

        for i in range(new_candidate.shape[0]):
            new_candidate[i] = self.random_state.uniform(self.lower_bound[i], self.upper_bound[i])

        return new_candidate

