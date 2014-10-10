from apsis import OptimizationCoreInterface, RandomSearchCore
from apsis.OptimizationCoreInterface import OptimizationCoreInterface

optimizers_available = [RandomSearchCore]


def check_optimizer(self, optimizer):
    if isinstance(optimizer, OptimizationCoreInterface):
        return optimizer

    index = [X.__name__ for X in self.optimizers_available].find(optimizer)

    if index != -1:
        return self.optimizers_available[index]
    else:
        raise ValueError("Optimizer " + optimizer + "not found")

