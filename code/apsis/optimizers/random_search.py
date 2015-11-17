__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate


class RandomSearch(Optimizer):
    """
    This is a simple random search implementation.

    It has the advantage of allowing highly parallel optimization, but only
    limited performance. It supports every available parameter type.

    Attributes
    ----------
    random_state : randomstate, optional
        The (optional) random state to use. See numpy random states.

    """
    SUPPORTED_PARAM_TYPES = [NominalParamDef, NumericParamDef]

    random_state = None
    logger = None

    def __init__(self, experiment, optimizer_params=None):
        """
        Initializes the random search optimizer.

        Parameters
        ----------
        experiment : Experiment
            The experiment representing the current state of the execution.
        optimizer_params : dict, optional
            Dictionary of the optimizer parameters. If None, some standard
            parameters will be assumed.
            Available parameters are
            "random_state" : randomstate, optional
                The random state to use. See numpy random states.

        Raises
        ------
        ValueError
            Iff the experiment is not supported.
        """
        if optimizer_params is None:
            optimizer_params = {}
        self.random_state = optimizer_params.get("random_state", None)
        Optimizer.__init__(self, experiment, optimizer_params)

    def get_next_candidates(self, num_candidates=1):
        candidate_list = []
        for i in range(num_candidates):
            candidate_list.append(self._gen_one_candidate())
        return candidate_list

    def _gen_one_candidate(self):
        """
        Generates a single candidate.

        This is done by generating parameter values for each of the
        available parameters.

        Returns
        -------
        candidate : Candidate
            The generated candidate
        """
        self.random_state = check_random_state(self.random_state)
        value_dict = {}
        for key, param_def in self._experiment.parameter_definitions.iteritems():
            value_dict[key] = self._gen_param_val(param_def)
        return Candidate(value_dict)

    def _gen_param_val(self, param_def):
        """
        Returns a random parameter value for param_def.

        This is done by generating warped_size many different 0-1 values, which
        are then warped out.

        Parameters
        ----------
        param_def : ParamDef
            The parameter definition from which to choose one at random.
        Returns
        -------
        param_val:
            The generated parameter value.
        """
        return param_def.warp_out(list(self.random_state.uniform(0, 1, param_def.warped_size())))
