__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from abc import ABCMeta, abstractmethod


class Optimizer(object):
    """
    This defines a basic optimizer interface.
    """
    __metaclass__ = ABCMeta

    SUPPORTED_PARAM_TYPES = []

    @abstractmethod
    def __init__(self, optimizer_params):
        """
        Initializes the optimizer with the arguments under optimizer_params.

        Parameters
        ----------
        optimizer_arguments : dict
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
        experiment : Experiment
            The experiment to form the base of the next candidate.

        Returns
        -------
        next_candidate : list of Candidate
            The Candidate to next evaluate.
        """
        pass

    def _is_experiment_supported(self, experiment):
        """
        Tests whether all parameter types in experiment are supported by this
        optimizer.

        Parameters
        ----------
        experiment : Experiment
            The experiment to test.

        Returns
        -------
        supported : bool
            False iff one or more of experiment's parameter definitions are not
            supported.
        """
        for name, pd in experiment.parameter_definitions.iteritems():
            if not self._is_supported_param_type(pd):
                return False
        return True

    def _is_supported_param_type(self, param):
        """
        Tests whether a certain parameter is supported by the optimizer.

        Parameters
        ----------
        param :
            The parameter to be tested

        Result
        ------
        is_supported : bool
            True iff param is supported by this optimizer.
        """
        if isinstance(self.SUPPORTED_PARAM_TYPES, list):
            for sup in self.SUPPORTED_PARAM_TYPES:
                if isinstance(param, sup):
                    return True

        return False