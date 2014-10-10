from apsis.RandomSearchCore import RandomSearchCore
import numpy as np

lower_bound = np.zeros((3,1))
upper_bound = np.ones((3,1))

randomSearchCore = RandomSearchCore(lower_bound, upper_bound)

