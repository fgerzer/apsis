__author__ = 'Frederik Diehl'

from apsis.optimizers.random_search import RandomSearch
from apsis.optimizers.optimizer import Optimizer, QueueBasedOptimizer
from apsis.optimizers.bayesian_optimization import BayesianOptimizer
import numpy as np

AVAILABLE_OPTIMIZERS = {"RandomSearch": RandomSearch,
                        "BayOpt": BayesianOptimizer}

def check_optimizer(optimizer, experiment, optimizer_arguments=None):
    """
    Checks whether optimizer is an optimizer or builds one.

    Specifically, it tests whether optimizer is an Optimizer instance. If
    it is, it is returned unchanged, all other parameters are ignored. If
    it is a class of optimizer, it will initialize it with experiment and
    optimizer_arguments. If it is a basestring, it will be translated via
    optimizer_utils.AVAILABLE_OPTIMIZERS, then initialized.

    Parameters
    ----------
    optimizer : string, Optimizer instance, optimizer class
        The optimizer to initialize. If optimizer instance, the other
        parameters will be ignored.
    experiment : Experiment
        The experiment defining the optimizer.
    optimizer_arguments : dict, optional
        The parameters governing the behaviour of the optimizer. If None,
        default values are used.
        This class introduces an additional parameter, called multiprocessing.
        If "queue", the default, it will initialize the optimizer abstracted by
        a QueueBasedOptimizer. If "none", it will initialize it directly.

    Returns
    -------
    optimizer : Optimizer instance
        An initialized optimizer instance.

    Raises
    ------
    ValueError
        If the optimizer is a string, and one cannot find it in
        AVAILABLE_OPTIMIZERS. If not an optimizer subclass. If the
        multiprocessing argument is not an acceptable value.

    """
    if optimizer_arguments is None:
        optimizer_arguments = {}
    multi_architecture = optimizer_arguments.get("multiprocessing", "queue")
    if isinstance(optimizer, Optimizer):
        return optimizer

    if isinstance(optimizer, basestring):
        try:
            optimizer = AVAILABLE_OPTIMIZERS[optimizer]
        except:
            raise ValueError("No corresponding optimizer found for %s. "
                             "Optimizer must be in %s" %(
                str(optimizer), AVAILABLE_OPTIMIZERS.keys()))

    if not issubclass(optimizer, Optimizer):
        raise ValueError("%s is of type %s, not Optimizer type."
                         %(optimizer, type(optimizer)))

    if multi_architecture == "queue":
        return QueueBasedOptimizer(optimizer, experiment, optimizer_arguments)
    elif multi_architecture == "none":
        return optimizer(experiment, optimizer_arguments)
    else:
        raise ValueError("%s is not supported as a multi-architecture "
                         "parameter. Currently supported are %s" %(
            multi_architecture, ["none", "queue"]))