__author__ = 'Frederik Diehl'

from apsis.utilities.benchmark_functions import *
import random

class testBenchmarkFunctions(object):

    def test_branin_func(self):
        x = random.uniform(-5, 10)
        y = random.uniform(0, 15)
        result = branin_func(x, y)

    def test_gen_noise(self):
        dims = 5
        points = 5
        noise_gen = gen_noise(dims, points)
        x = [0.5, 0.5, 0.5, 0.5, 0.5]
        val = get_noise_value_at(x, 0.5, noise_gen)
