from apsis.utilities.PreComputedGrid import PreComputedGrid
from apsis.evaluation.EvaluationFramework import EvaluationFramework
from apsis.models.ParamInformation import *
from apsis.RandomSearchCore import RandomSearchCore
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
import math
import logging

def obj_func(candidate):
    return math.sin(candidate.params[0])

logging.basicConfig(level=logging.DEBUG)
func = obj_func

grid = PreComputedGrid()
param_defs = [LowerUpperNumericParamDef(0., 10.)]
grid.build_grid_points(param_defs=param_defs, dimensionality=1000)
grid.precompute_results(func)
ev = EvaluationFramework()

optimizers = [RandomSearchCore({"param_defs": param_defs}),
              SimpleBayesianOptimizationCore({"param_defs": param_defs})]
steps = 50
ev.plot_precomputed_grid(optimizers, ["random", "bayes"], grid, steps)

raw_input("FINISHED")