__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate

class RandomSearch(Optimizer):
    SUPPORTED_PARAM_TYPES = [NominalParamDef, NumericParamDef]

    random_state = None

    def __init__(self, optimizer_params, out_queue, in_queue, min_candidates=1):
        if optimizer_params is None:
            optimizer_params = {}
        self.random_state = optimizer_params.get("random_state", None)
        Optimizer.__init__(self, optimizer_params, out_queue, in_queue, min_candidates)

    def _gen_candidates(self, num_candidates=1):
        list = []
        for i in range(num_candidates):
            list.append(self._gen_one_candidate())
        return list

    def _gen_one_candidate(self):
        self.random_state = check_random_state(self.random_state)
        value_dict = {}
        for key, param_def in self._experiment.parameter_definitions.iteritems():
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

    def _refit(self):
        pass