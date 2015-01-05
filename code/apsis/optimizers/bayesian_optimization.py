__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.random_search import RandomSearch
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate
from apsis.optimizers.bayesian.acquisition_functions import *
import GPy
import logging

class SimpleBayesianOptimizer(Optimizer):
    """
    This implements a simple bayesian optimizer.

    It is simple because it only implements the simplest form - no freeze-thaw,
    (currently) now multiple workers, only numeric parameters.

    Attributes
    ----------
    kernel: GPy Kernel
        The Kernel to be used with the gp. Note that this is currently not
        possible to be set from the outside.
    acquisition_function: acquisition_function
        The acquisition function to use
    acquisition_hyperparams:
        The acquisition hyperparameters.
    random_state: scipy random_state or int.

    """
    #TODO document attributes.

    SUPPORTED_PARAM_TYPES = [NumericParamDef, PositionParamDef]

    kernel = None
    acquisition_function = None
    acquisition_hyperparams = None

    random_state = None
    random_searcher = None

    gp = None
    initial_random_runs = 10
    num_gp_restarts = 10

    logger = None

    def __init__(self, optimizer_arguments=None):
        self.logger = logging.getLogger(__name__)
        #TODO documentation.
        if optimizer_arguments is None:
            optimizer_arguments = {}
        self.initial_random_runs = optimizer_arguments.get(
            'initial_random_runs', self.initial_random_runs)
        self.random_state = check_random_state(
            optimizer_arguments.get('random_state', None))
        self.acquisition_hyperparams = optimizer_arguments.get(
            'acquisition_hyperparams', None)
        self.num_gp_restarts = optimizer_arguments.get(
            'num_gp_restarts', self.num_gp_restarts)
        self.acquisition_function = optimizer_arguments.get(
            'acquisition', ExpectedImprovement)(self.acquisition_hyperparams)
        #TODO Find a better way to set the Kernel parameters, that is at all.
        self.random_searcher = RandomSearch({"random_state": self.random_state})

        self.num_precomputed = optimizer_arguments.get('num_precomputed', 0)

    def get_next_candidates(self, experiment):

        #check whether a random search is necessary.
        if len(experiment.candidates_finished) < self.initial_random_runs:
            return self.random_searcher.get_next_candidates(experiment)

        self._refit(experiment)
        #TODO refitted must be set, too.
        candidates = []
        new_candidate_points = self.acquisition_function.compute_proposals(
            self.gp, experiment, number_proposals=1, random_steps=1000)

        for point in new_candidate_points:

            point_candidate = Candidate(experiment.warp_pt_out(point))
            candidates.append(point_candidate)
        return candidates



    def _refit(self, experiment):
        candidate_matrix = np.zeros((len(experiment.candidates_finished),
                                     len(experiment.parameter_definitions)))
        results_vector = np.zeros((len(experiment.candidates_finished), 1))

        param_names = sorted(experiment.parameter_definitions.keys())
        self.kernel = GPy.kern.Matern52(len(param_names), ARD=True)
        for i, c in enumerate(experiment.candidates_finished):
            warped_in = experiment.warp_pt_in(c.params)
            param_values = []
            for pn in param_names:
                param_values.append(warped_in[pn])
            candidate_matrix[i, :] = param_values
            results_vector[i] = c.result

        self.logger.debug("Refitting gp with cand %s and results %s" %(candidate_matrix, results_vector))
        self.gp = GPy.models.GPRegression(candidate_matrix, results_vector, self.kernel)
        self.gp.constrain_positive("*")
        #self.gp.constrain_bounded('.*lengthscale*', 0.1, 1.)
        #self.gp.constrain_bounded('.*noise*', 0.1, 1.)
        self.gp.optimize_restarts(num_restarts=self.num_gp_restarts,
                                  verbose=False)