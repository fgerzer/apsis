from apsis.RandomSearchCore import RandomSearchCore
import numpy as np
import logging
import math
import nose.tools as nt


# noinspection PyPep8Naming,PyPep8Naming
class testRandomSearchCore():
    random_search_core = None

    def setUp(self):
        lower_bound = np.zeros((1, 2))
        upper_bound = np.ones((1, 2))
        param_dict = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "minimization_problem": True,
            "random_state": None
        }
        self.random_search_core = RandomSearchCore(param_dict)

    def test_initialization(self):
        lower_bound = np.zeros((1, 2))
        upper_bound = np.ones((1, 2))

        assert self.random_search_core is not None
        RandomSearchCore({"lower_bound": lower_bound, "upper_bound": upper_bound})
        RandomSearchCore({"lower_bound": [0, 0], "upper_bound": upper_bound})
        RandomSearchCore({"lower_bound": lower_bound, "upper_bound": [1, 1]})

    @nt.raises(ValueError)
    def test_upper_bound_not_filled(self):
        lower_bound = np.zeros((1, 1))
        RandomSearchCore({"lower_bound": lower_bound})

    @nt.raises(ValueError)
    def test_lower_bound_not_filled(self):
        upper_bound = np.zeros((1, 1))
        RandomSearchCore({"upper_bound": upper_bound})

    @nt.raises(ValueError)
    def test_lower_bigger_than_upper_bound(self):
        upper_bound = np.ones((1, 4))
        lower_bound = np.zeros((1, 4))
        lower_bound[0, 3] = 1
        RandomSearchCore({"upper_bound": upper_bound})


    def test_next_candidate(self):
        next_candidate = self.random_search_core.next_candidate()
        assert next_candidate is not None

    def test_working(self):
        candidate = self.random_search_core.next_candidate()
        continuing = self.random_search_core.working(candidate, status="finished")
        assert isinstance(continuing, bool)
        logging.info(__name__ + " results in " + str(continuing))

    def test_convergence_multiple_workers(self):
        self.random_search_core = RandomSearchCore({"lower_bound": [0], "upper_bound": [1]})
        f = math.sin
        cands = []
        best_result = None
        for i in range(100):
            cand = self.random_search_core.next_candidate()
            point = cand.params
            value = f(point)
            if best_result is None or value < best_result:
                best_result = value
            cand.result = value
            cands.append(cand)
        for i in range(100):
            assert not self.random_search_core.working(cands[i], "finished")
        nt.eq_(self.random_search_core.best_candidate.result, best_result,
                       str(self.random_search_core.best_candidate.result) + " != " + str(best_result))

    def test_convergence_one_worker(self):
        self.random_search_core = RandomSearchCore({"lower_bound": [0], "upper_bound": [1]})
        f = math.sin
        best_result = None
        for i in range(100):
            cand = self.random_search_core.next_candidate()
            point = cand.params
            value = f(point)
            if best_result is None or value < best_result:
                best_result = value

            cand.result = value
            assert not self.random_search_core.working(cand, "finished")
        nt.eq_(self.random_search_core.best_candidate.result, best_result,
                       str(self.random_search_core.best_candidate.result) + " != " + str(best_result))