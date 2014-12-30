__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod


class Optimizer(object):
    """
    This defines a basic optimizer interface.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_next_candidate(self, experiment):
        """
        Returns a Candidate object given an experiment.

        Parameters
        ----------
        experiment: Experiment
            The experiment to form the base of the next candidate.

        Returns
        -------
        next_candidate: Candidate
            The Candidate to next evaluate.
        """
        return None