__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.random_search import RandomSearch
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate
from apsis.optimizers.bayesian.acquisition_functions import *
import GPy


class BayesianOptimizer(Optimizer):
    """
    This is a bayesian optimizer class.

    It is a subclass of Optimizer, and internally uses GPy.
    Currently, it supports Numeric and PositionParamDefs, with support for
    NominalParamDef needing to be integrated.

    Attributes
    ----------
    SUPPORTED_PARAM_TYPES : list of ParamDefs
        The supported parameter types. Currently only numeric and position.
    kernel : GPy Kernel
        The Kernel to be used with the gp.
    acquisition_function : acquisition_function
        The acquisition function to use
    acquisition_hyperparams :
        The acquisition hyperparameters.
    random_state : scipy random_state or int.
        The scipy random state or object to initialize one. For reproduction.
    random_searcher : RandomSearch
        The random search instance used to generate the first
        initial_random_runs candidates.
    gp : GPy gaussian process
        The gaussian process used here.
    initial_random_runs : int
        The number of initial random runs before using the GP. Default is 10.
    num_gp_restarts : int
        GPy's optimization requires restarts to find a good solution. This
        parameter controls this. Default is 10.
    logger: logger
        The logger instance for this object.
    """
    SUPPORTED_PARAM_TYPES = [NumericParamDef, NominalParamDef]

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

    _experiment = None

    _logger = None

    def __init__(self, experiment, optimizer_params=None):
        """
        Initializes a bayesian optimizer.
        Parameters
        ----------
        experiment : Experiment
            The experiment for which to optimize.
        optimizer_arguments: dict of string keys
            Sets the possible arguments for this optimizer. Available are:
            "initial_random_runs" : int, optional
                The number of initial random runs before using the GP. Default
                is 10.
            "random_state" : scipy random state, optional
                The scipy random state or object to initialize one. Default is
                None.
            "acquisition_hyperparameters" : dict, optional
                dictionary of acquisition-function hyperparameters
            "num_gp_restarts" : int
                GPy's optimization requires restarts to find a good solution.
                This parameter controls this. Default is 10.
            "acquisition" : AcquisitionFunction
                The acquisition function to use. Default is
                ExpectedImprovement.
            "num_precomputed" : int
                The number of points that should be kept precomputed for faster
                multiple workers.
        """
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
        self.random_searcher = RandomSearch(experiment, optimizer_params)
        Optimizer.__init__(self, experiment, optimizer_params)

    def get_next_candidates(self, num_candidates=1):
        if len(self._experiment.candidates_finished) < self.initial_random_runs:
            # we do a random search.
            return self.random_searcher.get_next_candidates(num_candidates)
        candidates = []
        new_candidate_points = self.acquisition_function.compute_proposals(
            self.gp, self._experiment, number_proposals=num_candidates,
            return_max=self.return_max
        )
        self.return_max = False
        for point_and_value in new_candidate_points:
            # get the the candidate point which is the first entry in the tuple.
            point_candidate = Candidate(self._experiment.warp_pt_out(point_and_value[0]))
            candidates.append(point_candidate)
        return candidates

    def update(self, experiment):
        self._experiment = experiment
        if len(self._experiment.candidates_finished) < self.initial_random_runs:
            return
        self.return_max = True

        parameter_warped_size = 0
        for p in experiment.parameter_definitions.values():
            parameter_warped_size += p.warped_size()

        candidate_matrix = np.zeros((len(experiment.candidates_finished),
                                     parameter_warped_size))
        results_vector = np.zeros((len(experiment.candidates_finished), 1))

        param_names = sorted(experiment.parameter_definitions.keys())
        self.kernel = self._check_kernel(self.kernel, len(param_names),
                                         kernel_params=self.kernel_params)

        for i, c in enumerate(self._experiment.candidates_finished):
            warped_in = self._experiment.warp_pt_in(c.params)
            param_values = []
            for pn in param_names:
                param_values.extend(warped_in[pn])
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