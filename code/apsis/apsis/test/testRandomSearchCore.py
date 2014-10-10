from apsis.RandomSearchCore import RandomSearchCore
import numpy as np
import logging

# noinspection PyPep8Naming,PyPep8Naming
class testRandomSearchCore():
    random_search_core = None

    def setUp(self):
        lower_bound = np.zeros((3, 1))
        upper_bound = np.ones((3, 1))
        self.random_search_core = RandomSearchCore(lower_bound, upper_bound)

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