__author__ = 'Frederik Diehl'

from apsis.optimizers.random_search import RandomSearch
from apsis.optimizers.optimizer import Optimizer, QueueBasedOptimizer
from apsis.optimizers.bayesian_optimization import BayesianOptimizer

AVAILABLE_OPTIMIZERS = {"RandomSearch": RandomSearch, "BayOpt": BayesianOptimizer}

def check_optimizer(optimizer, experiment, optimizer_arguments=None):
    if optimizer_arguments is None:
        optimizer_arguments = {}
    queue_based = optimizer_arguments.get("queue_based", True)
    if isinstance(optimizer, Optimizer):
        return optimizer

    if isinstance(optimizer, basestring):
        try:
            optimizer = AVAILABLE_OPTIMIZERS[optimizer]
        except:
            raise ValueError("No corresponding optimizer found for %s. Optimizer must "
                     "be in %s" %(str(optimizer), AVAILABLE_OPTIMIZERS.keys()))

    if not type(optimizer) is Optimizer:
        pass
        #TODO raise ValueError for not being of optimizer type.
        #raise ValueError

    if queue_based:
        return QueueBasedOptimizer(optimizer, optimizer_arguments, experiment)
    else:
        return optimizer(optimizer_arguments, experiment)

