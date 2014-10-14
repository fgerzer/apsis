#!/usr/bin/python
from abc import ABCMeta, abstractmethod

from apsis.models import Candidate


class OptimizationCoreInterface(object):
    """
    The interface definition all optimizers have to fulfill.
    """
    __metaclass__ = ABCMeta

    minimization = True

    @abstractmethod
    def __init__(self, params):
        """
        Init method to initialize optimization core.

        :param params: A dictionary of parameters for the optimizer.
        Should contain at least upper and lower bound.
        For more arguments (which may be required by the individual optimizers)
        see their documentation.
        :return: this object.
        """
        pass

    @abstractmethod
    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points that
        need to be evaluated.

        :param worker_id: A string id for the worker calling this method.
        :return: An object of type Candidate that contains information related
        to the point that shall be evaluated.
        """
        pass

    @abstractmethod
    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        """
        Method that is used by used by workers to announce their currently
        executed points.

        The working method is the main reply method for workers. It is used to
        announce to the optimizer that they are working on a point; to return
        points and to announce that they are paused. It also allows for an
        optimizer to kill workers.
        Later, workers may be required by a core to regularly send a working
        message.

        :param candidate: an object of type Candidate that contains information
        related to the point the worker is
        currently evaluating
        :param status: a string containing either of the values ('finished',
        'working', 'pausing').
        A worker sending 'finished' indicates that it stops the evaluation.
        Sending 'working' indicates that the worker
        wants to go on working on this point. 'pausing' indicates that the
        worker needed to pause for some external
        reason and continuing on evaluating this point should be done by
        another worker.
        :param worker_id: A string id for the worker calling this method.
        Unused by default.
        :param can_be_killed: A boolean stating if the worker can be killed
        from the core or not.
        :return: a boolean to tell the worker if it should continue or stop
        the evaluation
        """
        pass

    def is_better_candidate_as(self, one, two):
        """
        Tests whether one is a better Candidate than two.

        This is dependant on whether the problem is one of minimization or
        maximization.
        It is done by comparing their results.
        :param one: Candidate that should be better.
        :param two: Candidate that acts as a baseline.
        :return: True iff one is better than two.
        """
        if not isinstance(one, Candidate):
            raise ValueError("Value is not a candidate " + str(one)
                             + " but a " + str(type(one)))
        if not isinstance(two, Candidate):
            raise ValueError("Value is not a candidate " + str(two)
                             + " but a " + str(type(two)))

        if self.minimization:
            return one.result < two.result
        else:
            return one.result > two.result
