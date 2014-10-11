from apsis import RandomSearchCore
from apsis.OptimizationCoreInterface import OptimizationCoreInterface

optimizers_available = [RandomSearchCore]


def check_optimizer(optimizer):
    if isinstance(optimizer, OptimizationCoreInterface):
        return optimizer

    index = [X.__name__ for X in optimizers_available].find(optimizer)

    if index != -1:
        return optimizers_available[index]
    else:
        raise ValueError("Optimizer " + optimizer + "not found")

