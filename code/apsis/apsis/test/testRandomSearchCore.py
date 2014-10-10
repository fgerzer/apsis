from apsis.RandomSearchCore import RandomSearchCore
import numpy as np

from nose import with_setup

class testRandomSearchCore():
    random_search_core = None

    def setUp(self):
        lower_bound = np.zeros((3, 1))
        upper_bound = np.ones((3, 1))
        self.random_search_core = RandomSearchCore(lower_bound, upper_bound)
        print("Random init")


    def test_initialization(self):
        assert self.random_search_core is not None

    def test_next_candidate(self):
        next_candidate = self.random_search_core.next_candidate()

        print(str(next_candidate))
