from apsis.RandomSearchCore import RandomSearchCore
import numpy as np
import logging
import math
import time
import nose.tools

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

    def test_convergence_multiple_workers(self):
        f = math.sin
        cands = []
        best_result = None
        for i in range(100):
            cand = self.random_search_core.next_candidate()
            point = cand.params
            value = f(point)
            if (best_result is None or value < best_result):
                best_result = value
            cand.result = value
            cands.append(cand)
        for i in range(100):
            assert self.random_search_core.working(cands[i], "finished") == False
        nose.tools.eq_(self.random_search_core.best_candidate.result, best_result,
                       str(self.random_search_core.best_candidate.result) + " != " + str(best_result))

    def test_convergence_one_worker(self):
        f = math.sin
        best_result = None
        for i in range(100):
            cand = self.random_search_core.next_candidate()
            point = cand.params
            value = f(point)
            if (best_result is None or value < best_result):
                best_result = value

            cand.result = value
            assert self.random_search_core.working(cand, "finished") == False
        nose.tools.eq_(self.random_search_core.best_candidate.result, best_result,
                       str(self.random_search_core.best_candidate.result) + " != " + str(best_result))