__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.random_search import RandomSearch
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate
from apsis.optimizers.bayesian.acquisition_functions import *
from apsis.utilities.import_utils import import_if_exists
import GPy
import logging

mcmc_imported, pm = import_if_exists("pymcmc")

class BayesianOptimizer(Optimizer):
    SUPPORTED_PARAM_TYPES = [NumericParamDef, PositionParamDef]

    kernel = None
    kernel_params = None
    acquisition_function = None
    acquisition_hyperparams = None

    random_state = None
    random_searcher = None

    gp = None
    initial_random_runs = 10
    num_gp_restarts = 10

    return_max = True

    _logger = None

    def __init__(self, optimizer_params, out_queue, in_queue, min_candidates=5):
        self._logger = get_logger(self)
        if optimizer_params is None:
            optimizer_params = {}
        self.random_state = optimizer_params.get("random_state", None)

        self.initial_random_runs = optimizer_params.get(
            'initial_random_runs', self.initial_random_runs)
        self.random_state = check_random_state(
            optimizer_params.get('random_state', None))
        self.acquisition_hyperparams = optimizer_params.get(
            'acquisition_hyperparams', None)
        self.num_gp_restarts = optimizer_params.get(
            'num_gp_restarts', self.num_gp_restarts)
        if not isinstance(optimizer_params.get('acquisition'), AcquisitionFunction):
            self.acquisition_function = optimizer_params.get(
                'acquisition', ExpectedImprovement)(self.acquisition_hyperparams)
        else:
            self.acquisition_function = optimizer_params.get("acquisition")
        self.kernel_params = optimizer_params.get("kernel_params", {})
        self.kernel = optimizer_params.get("kernel", "matern52")
        Optimizer.__init__(self, optimizer_params, out_queue, in_queue, min_candidates)

    def _gen_candidates(self, num_candidates=1):

        if len(self._experiment.candidates_finished) < self.initial_random_runs:
            #we do a random search.
            return self._gen_candidates_randomly(num_candidates)
        candidates = []
        new_candidate_points = self.acquisition_function.compute_proposals(
            self.gp, self._experiment, number_proposals=num_candidates,
            return_max=self.return_max
        )
        self.return_max = False
        for point_and_value in new_candidate_points:
            #get the the candidate point which is the first entry in the tuple.
            point_candidate = Candidate(self._experiment.warp_pt_out(point_and_value[0]))
            candidates.append(point_candidate)
        return candidates

    def _gen_candidates_randomly(self, num_candidates=1):
        list = []
        for i in range(num_candidates):
            list.append(self._gen_one_candidate_randomly())
        return list


    def _gen_one_candidate_randomly(self):
        self.random_state = check_random_state(self.random_state)
        value_dict = {}
        for key, param_def in self._experiment.parameter_definitions.iteritems():
            value_dict[key] = self._gen_param_val_randomly(param_def)
        return Candidate(value_dict)

    def _gen_param_val_randomly(self, param_def):
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
        self.return_max = True


        candidate_matrix = np.zeros((len(self._experiment.candidates_finished),
                                     len(self._experiment.parameter_definitions)))
        results_vector = np.zeros((len(self._experiment.candidates_finished), 1))


        param_names = sorted(self._experiment.parameter_definitions.keys())
        self.kernel = self._check_kernel(self.kernel, len(param_names),
                                         kernel_params=self.kernel_params)

        for i, c in enumerate(self._experiment.candidates_finished):
            warped_in = self._experiment.warp_pt_in(c.params)
            param_values = []
            for pn in param_names:
                param_values.append(warped_in[pn])
            candidate_matrix[i, :] = param_values
            results_vector[i] = c.result


        self._logger.debug("Refitting gp with cand %s and results %s"
                          %(candidate_matrix, results_vector))
        self.gp = GPy.models.GPRegression(candidate_matrix, results_vector,
                                          self.kernel)
        self.gp.constrain_positive("*")
        self.gp.constrain_bounded(0.1, 1, warning=False)
        self.gp.optimize_restarts(num_restarts=self.num_gp_restarts,
                                  verbose=False)



    def _check_kernel(self, kernel, dimension, kernel_params):
        """
        Checks and initializes a kernel.

        Parameters
        ----------
        kernel : kernel or string representation
            The kernel to use. If a kernel, is returned like that. If not, a
            new kernel is initialized with the respective parameters.
        dimension : int
            The dimensions of the new kernel.
        kernel_params : dict
            The dictionary of kernel parameters. Currently supported:
            "ARD" : bool, optional
                Whether to use ARD. Default is True.

        Returns
        -------
        kernel : GPy.kern
            A GPy kernel.
        """
        if (isinstance(kernel, GPy.kern.Kern)):
            return kernel
        translation_dict = {
            "matern52": GPy.kern.Matern52,
            "rbf": GPy.kern.RBF
        }

        if isinstance(kernel, unicode):
            kernel = str(kernel)

        if isinstance(kernel, str) and kernel in translation_dict:
            if kernel_params.get('ARD', None) is None:
                kernel_params['ARD'] = True

            constructed_kernel = translation_dict[kernel](dimension, **kernel_params)
            return constructed_kernel

        raise ValueError("%s is not a kernel or string representing one!" %kernel)