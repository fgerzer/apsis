from apsis.RandomSearchCore import RandomSearchCore
import numpy as np

from nose import with_setup

lower_bound = np.zeros((3, 1))
upper_bound = np.ones((3, 1))

def setup_func():
    "set up test fixtures"

def teardown_func():
    "tear down test fixtures"

@with_setup(setup_func, teardown_func)
def test_initialization():
    assert RandomSearchCore(lower_bound, upper_bound) is not None
