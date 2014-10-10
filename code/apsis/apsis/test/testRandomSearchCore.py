from apsis.RandomSearchCore import RandomSearchCore
import numpy as np

from nose import with_setup

random_search_core = None

def setup_func():
    "set up test fixtures"

def teardown_func():
    "tear down test fixtures"

def random_init():
    lower_bound = np.zeros((3,1))
    upper_bound = np.ones((3,1))
    global random_search_core
    random_search_core = RandomSearchCore(lower_bound, upper_bound)


def test_initialization():
    lower_bound = np.zeros((3,1))
    upper_bound = np.ones((3,1))
    assert RandomSearchCore(lower_bound, upper_bound) is not None

@with_setup(random_init)
def test_next_candidate():
    next_candidate = random_search_core.next_candidate()

    print(str(next_candidate))
