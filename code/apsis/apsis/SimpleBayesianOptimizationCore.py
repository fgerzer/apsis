from apsis.OptimizationCoreInterface import OptimizationCoreInterface, \
    ListBasedCore
from apsis.RandomSearchCore import RandomSearchCore
from apsis.models.Candidate import Candidate
from apsis.models.ParamInformation import NumericParamDef, PositionParamDef
from apsis.bayesian.AcquisitionFunctions import ExpectedImprovement
import numpy as np
import GPy
import logging
from apsis.utilities.randomization import check_random_state

class SimpleBayesianOptimizationCore(ListBasedCore):
    """
    This implements a simple bayesian optimizer.

    It is simple because it only implements the simplest form - no freeze-thaw,
    (currently) now multiple workers, only numeric parameters.
    """
    SUPPORTED_PARAM_TYPES = [NumericParamDef, PositionParamDef]

    kernel = None
    acquisition_function = None
    acquisition_hyperparams = None

    random_state = None
    random_searcher = None

    num_precomputed = None
    just_refitted = True
    refit_necessary = False
    refit_running = False

    gp = None

    initial_random_runs = 10
    num_gp_restarts = 10

    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        super(SimpleBayesianOptimizationCore, self).working(
            candidate, status, worker_id, can_be_killed)
        logging.debug("Worker " + str(worker_id) + " informed me about work "
                                                   "in status " + str(status)
                      + "on candidate " + str(candidate))

        # first check if this point is known. if it is in finished, tell the
        # worker to kill the computation.
        if not self.transfer_to_working(candidate):
            return False

        # if finished remove from working and put to finished list.
        if status == "finished":
            self.deal_with_finished(candidate)

            # invoke the refitting
            if len(self.finished_candidates) >= self.initial_random_runs:
                self.refit_necessary = True
                self._check_refit_gp()
            return False

        elif status == "working":
            self.working_candidates.remove(candidate)
            self.working_candidates.append(candidate)
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
        super(SimpleBayesianOptimizationCore, self).__init__(params)
        if params.get('param_defs', None) is None:
            raise ValueError("Parameter definition list is missing!")
        if not self._is_all_supported_param_types(params["param_defs"]):
            raise ValueError(
                "Param list contains parameters of unsopported types. "
                "Supported types are  " + str(self.SUPPORTED_PARAM_TYPES))

        self.param_defs = params["param_defs"]
        self.minimization = params.get('minimization', True)
        self.initial_random_runs = params.get('initial_random_runs',
                                              self.initial_random_runs)
        self.random_state = check_random_state(params.get('random_state', None))
        self.acquisition_hyperparams = params.get('acquisition_hyperparams',
                                                  None)
        self.acquisition_function = params.get('acquisition',
                                               ExpectedImprovement)(
            self.acquisition_hyperparams)
        self.num_gp_restarts = params.get('num_gp_restarts',
                                          self.num_gp_restarts)

        dimensions = len(self.param_defs)
        self.kernel = params.get('kernel',
                                 GPy.kern.rbf)(dimensions, ARD=True)
        logging.debug("Kernel input dim " + str(self.kernel.input_dim))
        logging.debug("Kernel %s", str(self.kernel))
        self.random_searcher = RandomSearchCore({'param_defs': self.param_defs,
                                            "random_state": self.random_state})
        for i in range(self.initial_random_runs):
            self.pending_candidates.append(self.random_searcher.
                                           next_candidate())

        self.num_precomputed = params.get('num_precomputed', 0)

    def next_candidate(self, worker_id=None):
        # either we have pending candidates
        if len(self.pending_candidates) > 0:
            new_candidate = self.pending_candidates.pop(0)

            logging.debug("Core providing pending candidate "
                          + str(new_candidate))

        # or we need to generate new ones, which either includes random points
        elif len(self.finished_candidates) < self.initial_random_runs:
            new_candidate = self.random_searcher.next_candidate()
            logging.debug("Core providing new randomly generated candidate " +
                            str(new_candidate))
        else:
            acquisition_params = {'param_defs': self.param_defs,
                                  'gp': self.gp,
                                  'cur_max': self.best_candidate.result,
                                  "minimization": self.minimization
            }

            logging.debug("Running acquisition with args %s",
                          str(acquisition_params))

            new_candidate_points = self.acquisition_function.compute_proposal(
                acquisition_params, refitted=self.just_refitted,
                number_proposals=self.num_precomputed+1)

            for point in new_candidate_points:
                for i in range(len(point)):
                    point[i] = self.param_defs[i].warp_out(
                        point[i]
                    )
                point_candidate = Candidate(point)

                self.pending_candidates.append(point_candidate)

            new_candidate = self.pending_candidates.pop(0)
            self.just_refitted = False

        # add candidate to working list
        self.working_candidates.append(new_candidate)

        return new_candidate

    def _check_refit_gp(self):
        if self.refit_necessary and not self.refit_running:
            # replace by python mutex
            self.refit_running = True
            # TODO start new thread for refitting
            self._refit_gp()
            self.refit_running = False

    def _refit_gp(self):
        self.refit_necessary = False

        #empty the pendings because they differ after refitting
        self.pending_candidates = []
        self.just_refitted = True

        candidate_matrix = np.zeros((len(self.finished_candidates),
                                     len(self.param_defs)))
        results_vector = np.zeros((len(self.finished_candidates), 1))

        for i in range(len(self.finished_candidates)):
            candidate_matrix[i, :] = self._warp_in(
                self.finished_candidates[i].as_vector()
            )
            results_vector[i] = self.finished_candidates[i].result

        logging.debug("refitting gp with values %s and results %s",
                      str(candidate_matrix), str(results_vector))

        self.gp = GPy.models.GPRegression(candidate_matrix, results_vector,
                                          self.kernel)
        self.gp.constrain_bounded('.*lengthscale*', 0.1, 1.)
        self.gp.constrain_bounded('.*noise*', 0.1, 1.)
        logging.debug("Generated gp model. Refitting now.")
        # ensure restart
        self.gp.optimize_restarts(num_restarts=self.num_gp_restarts,
                                  verbose=False)
        logging.debug("Finished generating model.")

    def _warp_in(self, param_vector):
        """
        Warps the parameter vector, using the warp_in function for each
        self.param_defs.
        """
        logging.debug("Param_vector: %s" % str(param_vector))
        for i in range(len(param_vector)):
            logging.debug("Warping in. %f to %f" % (param_vector[i],
                                self.param_defs[i].warp_in(param_vector[i])))
            param_vector[i] = self.param_defs[i].warp_in(
                param_vector[i]
            )
        logging.debug("New param_vector: %s" % str(param_vector))
        return param_vector