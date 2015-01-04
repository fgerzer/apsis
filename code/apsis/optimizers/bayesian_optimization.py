__author__ = 'Frederik Diehl'

from apsis.optimizers.optimizer import Optimizer
from apsis.optimizers.random_search import RandomSearch
from apsis.models.parameter_definition import *
from apsis.utilities.randomization import check_random_state
from apsis.models.candidate import Candidate
from apsis.optimizers.bayesian.acquisition_functions import *
import GPy

class SimpleBayesianOptimizer(Optimizer):
    """
    This implements a simple bayesian optimizer.

    It is simple because it only implements the simplest form - no freeze-thaw,
    (currently) now multiple workers, only numeric parameters.
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

    def __init__(self, optimizer_arguments=None):
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
        self.random_searcher = RandomSearch(self.random_state)
        self.num_precomputed = optimizer_arguments.get('num_precomputed', 0)




    def get_next_candidate(self, experiment):
        self._refit(experiment)

        #check whether a random search is necessary.
        if len(experiment.candidates_finished) < self.initial_random_runs:
            return self.random_searcher.get_next_candidates(experiment)

        acquisition_params = {'gp': self.gp,
                              'cur_max': experiment.best_candidate.result,
                              "minimization": experiment.minimization_problem,
                              'random_search_steps': 100
        }
        #TODO refitted must be set, too.
        candidates = []
        new_candidate_points = self.acquisition_function.compute_proposal(
                acquisition_params, refitted=True,
                number_proposals=self.num_precomputed+1)


        for point in new_candidate_points:
            experiment.warp_pt_out(point)
            point_candidate = Candidate(point)
            candidates.append(point_candidate)
        return candidates



    def _refit(self, experiment):

        self.kernel = optimizer_arguments.get('kernel',
             GPy.kern.rbf)(dimensions, ARD=True)