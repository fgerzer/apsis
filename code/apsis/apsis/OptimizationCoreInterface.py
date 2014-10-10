#!/usr/bin/python
from abc import ABCMeta, abstractmethod


class OptimizationCoreInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, lower_bound, upper_bound):
        """
        Init method to initialize optimization core.
        :param params_lower_bound: Lower bound of optimization search space. Expected is a numpy vector of floats,
        one per dimension of the parameter space.
        :param params_upper_bound: Upper bound of optimization search space. Expected is a numpy vector of floats,
        one per dimension of the parameter space.
        :return: this object.
        """
        pass

    @abstractmethod
    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points that need to be evaluated.
        :param worker_id: A string id for the worker calling this method.
        :return: An object of type Candidate that contains information related to the point that shall be evaluated.
        """
        pass

    @abstractmethod
    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        """
        Method that is used by used by workers to
        :param candidate: an object of type Candidate that contains information related to the point the worker is
        currently evaluating
        :param status: a string containing either of the values ('finished', 'working', 'paused').
        A worker sending 'finished' indicates that it stops the evaluation. Sending 'working' indicates that the worker
        wants to go on working on this point. 'paused' indicates that the worker needed to pause for some external
        reason and continuing on evaluating this point should be done by another worker.
        :param worker_id: A string id for the worker calling this method. Unused by default.
        :param can_be_killed: A boolean stating if the worker can be killed from the core or not.
        :return: a boolean to tell the worker if it should continue or stop the evaluation
        """
        pass