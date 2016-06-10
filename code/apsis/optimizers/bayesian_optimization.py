__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.random_search import RandomSearch
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate
from apsis.optimizers.bayesian.acquisition_functions import *
from apsis.utilities.acquisition_utils import check_acquisition
import GPy
import apsis.utilities.acquisition_utils as acq_utils


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

    name = "BayOpt"
    return_max = True

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
        self._logger.debug("Initializing bayesian optimizer. Experiment is %s,"
                           " optimizer_params %s", experiment,
                           optimizer_params)
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

        self._logger.debug("Initialized relevant parameters. "
                           "initial_random_runs is %s, random_state is %s, "
                           "acquisition_hyperparams %s, num_gp_restarts %s",
                           self.initial_random_runs, self.random_state,
                           self.acquisition_hyperparams, self.num_gp_restarts)

        if not isinstance(optimizer_params.get('acquisition'),
                          AcquisitionFunction):
            self.acquisition_function = optimizer_params.get("acquisition",
                                                 ExpectedImprovement)
            self.acquisition_function = check_acquisition(
                acquisition=self.acquisition_function,
                acquisition_params=self.acquisition_hyperparams)
            self._logger.debug("acquisition is no AcquisitionFunction. Set "
                               "it to %s", self.acquisition_function)
        else:
            self.acquisition_function = optimizer_params.get("acquisition")
            self._logger.debug("Loaded acquisition function from "
                               "optimizer_params. Is %s",
                               self.acquisition_function)
        self.kernel_params = optimizer_params.get("kernel_params", {})
        self.kernel = optimizer_params.get("kernel", "matern52")

        self._logger.debug("Kernel details: Kernel is %s, kernel_params %s",
                           self.kernel, self.kernel_params)

        self.random_searcher = RandomSearch(experiment, optimizer_params)
        self._logger.debug("Initialized required RandomSearcher; is %s",
                           self.random_searcher)
        Optimizer.__init__(self, experiment, optimizer_params)
        self._logger.debug("Finished initializing bayOpt.")

    def get_next_candidates(self, num_candidates=1):
        self._logger.debug("Returning next %s candidates", num_candidates)
        if len(self._experiment.candidates_finished) < self.initial_random_runs:
            # we do a random search.
            random_candidates = self.random_searcher.get_next_candidates(
                num_candidates)
            self._logger.debug("Still in the random run phase. Returning %s",
                               random_candidates)
            return random_candidates
        candidates = []
        if self.gp is None:
            self._logger.debug("No gp available. Updating with %s",
                               self._experiment)
            self.update(self._experiment)

        new_candidate_points = self.acquisition_function.compute_proposals(
            self.gp, self._experiment, number_proposals=num_candidates,
            return_max=self.return_max
        )
        self._logger.debug("Generated new candidate points. Are %s",
                           new_candidate_points)
        self.return_max = False

        for point_and_value in new_candidate_points:
            # get the the candidate point which is the first entry in the tuple.
            point_candidate = Candidate(self._experiment.warp_pt_out(
                point_and_value[0]))
            candidates.append(point_candidate)
        self._logger.debug("Candidates extracted. Returning %s", candidates)
        return candidates

    def update(self, experiment):
        self._logger.debug("Updating bayOpt with %s", experiment)
        self._experiment = experiment
        if (len(self._experiment.candidates_finished) <
                self.initial_random_runs):
            self._logger.debug("Less than initial_random_runs. No refit "
                               "necessary.")
            return

        self.return_max = True

        candidate_matrix, results_vector = acq_utils.create_cand_matrix_vector(
            experiment, self.treat_failed)

        self.kernel = self._check_kernel(self.kernel, candidate_matrix.shape[1],
                                         kernel_params=self.kernel_params)
        self._logger.debug("Checked kernel. Kernel is %s", self.kernel)

        self._logger.log(5, "Refitting gp with cand %s and results %s"
                          %(candidate_matrix, results_vector))
        self.gp = GPy.models.GPRegression(candidate_matrix, results_vector,
                                          self.kernel)
        self.gp.constrain_positive("*")
        self.gp.constrain_bounded(0.1, 1, warning=False)
        self._logger.debug("Starting gp optimize.")
        self.gp.optimize_restarts(num_restarts=self.num_gp_restarts,
                                  verbose=False)
        self._logger.debug("gp optimize finished.")

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
        self._logger.debug("Checking kernel. Kernel is %s, dimension %s, "
                           "kernel_params %s", kernel, dimension,
                           kernel_params)
        if (isinstance(kernel, GPy.kern.Kern)):
            self._logger.debug("Already instance. No changes.")
            return kernel

        translation_dict = {
            "matern52": GPy.kern.Matern52,
            "rbf": GPy.kern.RBF
        }

        if isinstance(kernel, unicode):
            kernel = str(kernel)

        if isinstance(kernel, str) and kernel in translation_dict:
            self._logger.debug("Is string and can be translated. Kernel is %s",
                               kernel)
            if kernel_params.get('ARD', None) is None:
                self._logger.debug("ARD unknown, setting to True.")
                kernel_params['ARD'] = True

            constructed_kernel = translation_dict[kernel](dimension,
                                                          **kernel_params)
            self._logger.debug("Constructed kernel. Is %s", constructed_kernel)
            return constructed_kernel

        raise ValueError("%s is not a kernel or string representing one!"
                         %kernel)