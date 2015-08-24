__author__ = 'Frederik Diehl'

from apsis.optimizers.random_search import RandomSearch
from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.bayesian_optimization import BayesianOptimizer

AVAILABLE_OPTIMIZERS = {"RandomSearch": RandomSearch, "BayOpt": BayesianOptimizer}

def check_optimizer(optimizer, experiment, out_queue, in_queue, optimizer_arguments=None):
    #TODO documentation
    if isinstance(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, unicode):
        optimizer = str(optimizer)
    if isinstance(optimizer, str) and optimizer in AVAILABLE_OPTIMIZERS.keys():
        return AVAILABLE_OPTIMIZERS[optimizer](optimizer_arguments, experiment, out_queue, in_queue)
    raise ValueError("No corresponding optimizer found for %s. Optimizer must "
                     "be in %s" %(str(optimizer), AVAILABLE_OPTIMIZERS.keys()))