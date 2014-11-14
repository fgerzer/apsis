from apsis.utilities.PreComputedGrid import PreComputedGrid
from apsis.evaluation.EvaluationFramework import EvaluationFramework
from apsis.models.ParamInformation import *
from apsis.RandomSearchCore import RandomSearchCore
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
import math
import logging
import time

def test_plotting():
    def obj_func(candidate):
        x = candidate.params[0]
        y = candidate.params[1]
        return math.sin(x) * y**2

    logging.basicConfig(level=logging.DEBUG)
    func = obj_func

    grid = PreComputedGrid()
    param_defs = [LowerUpperNumericParamDef(0., 10.),
                  LowerUpperNumericParamDef(0., 10.)]
    grid.build_grid_points(param_defs=param_defs, dimensionality=1000)
    grid.precompute_results(func)
    ev = EvaluationFramework()

    optimizers = [RandomSearchCore({"param_defs": param_defs}),
                  SimpleBayesianOptimizationCore({"param_defs": param_defs,
                                                  "initial_random_runs": 5})]
    steps = 10
    ev.evaluate_and_plot_precomputed_grid(optimizers, ["random", "bayes"], grid, steps)

def grid_closeness():
    logging.basicConfig(level=logging.DEBUG)

    def obj_func(candidate):
        x = candidate.params[0]
        y = candidate.params[1]
        return math.sin(x) * y**2

    logging.basicConfig(level=logging.DEBUG)
    func = obj_func

    grid = PreComputedGrid()
    param_defs = [LowerUpperNumericParamDef(0., 1.),
                  LowerUpperNumericParamDef(0., 1.)]
    grid.build_grid_points(param_defs=param_defs, dimensionality=6)
    grid.precompute_results(func)
    for i in range(len(grid.grid_points)):
        logging.debug("%i: %s" %(i, grid.grid_points[i].params))
    logging.debug("Possible candidates: " + str(grid._get_possible_candidate_indices([0.1, 0.7], 0)))

test_plotting()