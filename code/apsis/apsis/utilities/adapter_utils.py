from apsis.RandomSearchCore import RandomSearchCore
from apsis.SimpleBayesianOptimizationCore import SimpleBayesianOptimizationCore
from apsis.OptimizationCoreInterface import OptimizationCoreInterface

optimizers_available = [RandomSearchCore, SimpleBayesianOptimizationCore]


def check_optimizer(optimizer):
    """
    Use to check and convert the argument to optimizer.

    The argument will be checked on whether it is an instance of
    OptimizerCoreInterface (in which case it will be returned) or a string
    representing one, in which case an Instance of it will be returned.

    Parameters
    ----------

    optimizer: string or OptimizationCoreInterface
        The optimizer or string to check.

    Returns
    -------

    optimizer: OptimizationCoreInterface
        An OptimizationCoreInterface fulfilling instance.

    Raises
    ------
    ValueError:
        If optimizer is not a subclass of OptimizationCoreInterface or a string
        representing one.
    """
    if isinstance(optimizer, OptimizationCoreInterface):
        return optimizer

    index = [X.__name__ for X in optimizers_available].index(optimizer)

    if index != -1:
        return optimizers_available[index]
    else:
        raise ValueError("Optimizer " + optimizer + "not found.")