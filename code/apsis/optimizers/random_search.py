__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate

class RandomSearch(Optimizer):
    """
    Implements a random searcher for parameter optimization.

    Attributes
    ----------
    random_state : None, int or random_state
        The random_state used to generate random numbers.
    """

    SUPPORTED_PARAM_TYPES = [NominalParamDef, NumericParamDef]

    random_state = None

    def __init__(self, optimizer_arguments=None):
        """
        Initializes the RandomSearch.

        Parameters
        ----------
        optimizer_arguments : dict or None, optional
            A dictionary of parameters for the optimizer. The following keys
            are used:
            "random_state" : random_state, None or int, optional
                A numpy random_state (after which it is modelled). Can be used
                for repeatability.
        """
        if optimizer_arguments is None:
            optimizer_arguments = {}
        self.random_state = optimizer_arguments.get("random_state", None)

    def get_next_candidates(self, experiment, num_candidates=1):
        list = []
        for i in range(num_candidates):
            list.append(self._get_one_candidate(experiment))
        return list

    def _get_one_candidate(self, experiment):
        self.random_state = check_random_state(self.random_state)
        value_dict = {}
        for key, param_def in experiment.parameter_definitions.iteritems():
            value_dict[key] = self._gen_param_val(param_def)
        return Candidate(value_dict)

    def _gen_param_val(self, param_def):
        """
        Returns a random parameter value for param_def.

        Parameters
        ----------
        param_def : ParamDef
            The parameter definition from which to choose one at random. The
            following may happen:
            NumericParamDef: warps_out a uniform 0-1 chosen value.
            NominalParamDef: chooses one of the values at random.

        Returns
        -------
        param_val:
            The generated parameter value.
        """
        if isinstance(param_def, NumericParamDef):
            return param_def.warp_out(self.random_state.uniform(0, 1))
        elif isinstance(param_def, NominalParamDef):
            return self.random_state.choice(param_def.values)
