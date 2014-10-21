from apsis.OptimizationCoreInterface import OptimizationCoreInterface, ListBasedCore
from apsis.RandomSearchCore import RandomSearchCore
from apsis.models.Candidate import Candidate
from apsis.models.ParamInformation import NumericParamDef
from apsis.bayesian.AcquisitionFunctions import ProbabilityOfImprovement
import numpy as np
import GPy
import logging


class SimpleBayesianOptimizationCore(OptimizationCoreInterface, ListBasedCore):
    SUPPORTED_PARAM_TYPES = [NumericParamDef]

    kernel = None
    acquisition_function = None

    gp = None

    initial_random_runs = 10
    num_gp_restarts = 10

    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        logging.debug("Worker " + str(worker_id) + " informed me about work "
                                                   "in status " + str(status)
                      + "on candidate " + str(candidate))

        # first check if this point is known
        self.perform_candidate_state_check(candidate)

        # if finished remove from working and put to finished list.
        if status == "finished":
            self.deal_with_finished(candidate)

            # invoke the refitting
            if len(self.finished_candidates) >= self.initial_random_runs:
                self._refit_gp()

            return False

        elif status == "working":
            # for now just continue working
            return True

        elif status == "pausing":
            self.working_candidates.remove(candidate)
            self.pending_candidates.append(candidate)

            return False

        else:
            logging.error("Worker " + worker_id + " posted candidate to core "
                                                  "with non correct status "
                                                  "value " + status)

        return True

    def __init__(self, params):
        if params.get('param_defs', None) is None:
            raise ValueError("Parameter definition list is missing!")

        # check if param_defs are supported
        if not self._is_all_supported_param_types(params["param_defs"]):
            raise ValueError(
                "Param list contains parameters of unsopported types. "
                "Supported types are  " + str(self.SUPPORTED_PARAM_TYPES))

        self.param_defs = params["param_defs"]

        self.initial_random_runs = params.get('initial_random_runs',
                                              self.initial_random_runs)

        self.finished_candidates = []
        self.working_candidates = []
        self.pending_candidates = []

        # either we have a given acquisition func or we use the standard one
        self.acquisition_function = params.get('acquisition',
                                               ProbabilityOfImprovement)()

        self.num_gp_restarts = params.get('num_gp_restarts',
                                          self.num_gp_restarts)

        # same for kernel
        dimensions = len(self.param_defs)
        logging.debug(dimensions)
        self.kernel = params.get('kernel',
                                 GPy.kern.rbf)(dimensions)

        logging.debug("Kernel input dim " + str(self.kernel.input_dim))
        logging.debug("Kernel %s", str(self.kernel))

        #generate random samples
        random_searcher = RandomSearchCore({'param_defs': self.param_defs})
        for i in range(self.initial_random_runs):
            self.pending_candidates.append(random_searcher.next_candidate())


    def next_candidate(self, worker_id=None):
        # either we have pending candidates
        if len(self.pending_candidates) > 0:
            new_candidate = self.pending_candidates.pop(0)

            logging.debug("Core providing pending candidate "
                          + str(new_candidate))

        # or we need to generate new ones
        else:
            acquisition_params = {'param_defs': self.param_defs,
                                  'gp': self.gp
                                  }

            logging.debug("Running acquisition with args %s", str(acquisition_params))
            new_candidate_point = self.acquisition_function.compute_max(
                acquisition_params)

            new_candidate = Candidate(new_candidate_point)

        # add candidate to working list
        self.working_candidates.append(new_candidate)

        return new_candidate

    def _refit_gp(self):
        candidate_matrix = np.zeros((len(self.finished_candidates),
                                     len(self.param_defs)))
        results_vector = np.zeros((len(self.finished_candidates), 1))

        for i in range(len(self.finished_candidates)):
            candidate_matrix[i, :] = self.finished_candidates[i].as_vector()
            results_vector[i] = self.finished_candidates[i].result

        logging.debug("refitting gp with values %s and results %s",
                      str(candidate_matrix), str(results_vector))

        self.gp = GPy.models.GPRegression(candidate_matrix, results_vector,
                                          self.kernel)

        #ensure restart
        self.gp.optimize_restarts(num_restarts=self.num_gp_restarts)



