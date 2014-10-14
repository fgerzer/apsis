from apsis.RandomSearchCore import RandomSearchCore
from apsis.OptimizationCoreInterface import OptimizationCoreInterface

optimizers_available = [RandomSearchCore]


def check_optimizer(optimizer):
    """
    Use to check and convert the argument to optimizer.

    The argument will be checked on whether it is an instance of
    OptimizerCoreInterface (in which case it will be returned) or a string
    representing one, in which case an Instance of it will be returned.
    :param optimizer: The optimizer or string to check.
    :return: An OptimizationCoreInterface fulfilling instance.
    :raise ValueError: If optimizer is not a subclass of
    OptimizationCoreInterface or a string representing one.
    """
    if isinstance(optimizer, OptimizationCoreInterface):
        return optimizer

    print(optimizers_available[0].__name__)
    index = [X.__name__ for X in optimizers_available].index(optimizer)

    if index != -1:
        return optimizers_available[index]
    else:
        raise ValueError("Optimizer " + optimizer + "not found")

