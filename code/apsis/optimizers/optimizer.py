__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod


class Optimizer(object):
    """
    This defines a basic optimizer interface.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, optimizer_params):
        """
        Initializes the optimizer with the arguments under optimizer_params.

        Parameters
        ----------
        optimizer_arguments: dict
            The parameters for the optimization. Depending on the optimizer,
            different arguments are needed.
        """
        pass

    @abstractmethod
    def get_next_candidates(self, experiment):
        """
        Returns several Candidate objects given an experiment.

        It is the free choice of the optimizer how many Candidates to provide,
        but it will provide at least one.
        Parameters
        ----------
        experiment: Experiment
            The experiment to form the base of the next candidate.

        Returns
        -------
        next_candidate: list of Candidate
            The Candidate to next evaluate.
        """
        pass