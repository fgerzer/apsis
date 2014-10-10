from apsis.RandomSearchCore import RandomSearchCore
import numpy as np
import logging
import math
import time

# noinspection PyPep8Naming,PyPep8Naming
class testRandomSearchCore():
    random_search_core = None

    def setUp(self):
        lower_bound = np.zeros((1, 1))
        upper_bound = np.ones((1, 1))
        self.random_search_core = RandomSearchCore(lower_bound, upper_bound, minimization_problem=True, random_state=None)

    def test_initialization(self):
        assert self.random_search_core is not None

    def test_next_candidate(self):
        next_candidate = self.random_search_core.next_candidate()
        assert next_candidate is not None

    def test_working(self):
        candidate = self.random_search_core.next_candidate()
        continuing = self.random_search_core.working(candidate, status = "finished")
        assert isinstance(continuing, bool)
        logging.info(__name__ + " results in " + str(continuing))

    def test_convergence(self):
        f = math.sin
        for i in range(100):
            cand = self.random_search_core.next_candidate()
            point = cand.params
            value = f(point)
            cand.result = value
            assert self.random_search_core.working(cand, "finished") == False
        print(self.random_search_core.best_candidate.result)