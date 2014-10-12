#!/usr/bin/python

from apsis.OptimizationCoreInterface import OptimizationCoreInterface
from apsis.Candidate import Candidate
from sklearn.utils import check_random_state
import numpy as np
import logging
from apsis.utilities.validation import check_array, \
    check_array_dimensions_equal


class RandomSearchCore(OptimizationCoreInterface):
    """
    Implements a random searcher for parameter optimization.
    """
    lower_bound = None
    upper_bound = None
    random_state = None

    finished_candidates = None
    working_candidates = None
    pending_candidates = None

    best_candidate = None

    def __init__(self, params):
        """
        Initializes the random search.

        :param params: The parameters with which the random search should be
        executed. Currently requires four:
            lower_bound: A numpy float vector, representing the lower bound
            for each attribute.
            upper_bound: A numpy float vector, representing the upper bound for
            each attribute. Has to be the same
             dimension as lower_bound.
            random_state: A numpy random state. Defaults to None, in which case
            a new one is generated.
            minimization_problem: Whether the problem is a minimization one
            (True) or a maximization one (False).
             Defaults to True.
        :raise ValueError: If params does not contain lower_bound or
        upper_bound, or attributes are assigned bad values.
        """
        if not "lower_bound" in params:
            raise ValueError("No lower_bound in params dictionary.")
        if not "upper_bound" in params:
            raise ValueError("No lower_bound in params dictionary.")
        self.lower_bound = check_array(params["lower_bound"])
        self.upper_bound = check_array(params["upper_bound"])

        if not check_array_dimensions_equal(self.lower_bound,
                                            self.upper_bound):
            raise ValueError("Arrays are not the same dimension: "
                             + str(self.lower_bound) + " and "
                             + str(self.upper_bound))
        if not (self.lower_bound < self.upper_bound).all():
            raise ValueError("Some elements of lower_bound are bigger or "
                             "equal than some elements of upper_bound. "
                             "lower_bound: " + str(self.lower_bound) +
                             ", upper_bound: " + str(self.upper_bound))

        logging.debug("Initializing Random Search Core for bounds..." +
                      str(self.lower_bound) + " and " + str(self.upper_bound))
        self.random_state = check_random_state(params.get("random_state",
                                                          None))
        self.minimization = params.get("minimization_problem", True)

        self.finished_candidates = []
        self.working_candidates = []
        self.pending_candidates = []

        super(RandomSearchCore, self).__init__(params)

    #TODO deal with the case that candidate point is the same but
    # objects do not equal
    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        """
        Right now, RandomSearchCore works like this:
        It ensures candidate is in the working_candidates list. If it is in
        the finished_candidates list,
         the worker is told to terminate execution.
        If the status is 'working', the worker may continue.
        If the status is 'pausing', candidate is returned to the
        pending_candidates list.
        If the status is 'finished', candidate is appended to the
        finished_candidates list, and possibly the best
         result updated. Of course, the worker is then told to stop.
        """
        logging.debug("Worker " + str(worker_id) + " informed me about work "
                                                   "in status " + str(status)
                      + "on candidate " + str(candidate))

        #first check if this point is known
        if candidate not in self.working_candidates:
            #work on a finished candidate is discarded and should abort
            if candidate in self.finished_candidates:
                return False
            #if work is carried out on candidate and it was pending it is no
            # longer pending
            elif candidate in self.pending_candidates:
                self.pending_candidates.remove(candidate)

            logging.debug("Candidate was UNKNOWN and not FINISHED "
                          + str(candidate)
                          + " Candidate added to WORKING list.")

            #but now it is a working item
            self.working_candidates.append(candidate)

        #if finished remove from working and put to finished list.
        if status == "finished":
            self.working_candidates.remove(candidate)
            self.finished_candidates.append(candidate)

            #check for the new best result
            if self.best_candidate is not None:
                if self.is_better_candidate_as(candidate, self.best_candidate):
                    logging.info("Cool - found new best candidate "
                                 + str(candidate))
                    self.best_candidate = candidate

            else:
                self.best_candidate = candidate

            return False
        elif status == "working":
            #for now just continue working
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

    def next_candidate(self, worker_id=None):
        """
        This method is invoked by workers to obtain next candidate points
        that need to be evaluated.

        The new Candidate object is return following these rules:
        If there are pending candidates, they are returned first. Otherwise
        a new random candidate is generated
        from a uniform distribution. It is made sure that this point has not
        been marked as finished yet.
        On return of a new candidate it is appended to the working_candidates
        list in this core.

        :param worker_id: not used here, but needed to satisfy interface of
        OptimizationCoreInterface.
        :return: an instance of a Candidate object that should be evaluated
        next.
        """
        #either we have pending candidates
        if len(self.pending_candidates) > 0:
            new_candidate = self.pending_candidates.pop(0)

            logging.debug("Core providing pending candidate "
                          + str(new_candidate))

        #or we need to generate new ones
        else:
            new_candidate_point = self._generate_new_random_vector()
            new_candidate = Candidate(new_candidate_point)

            while ((new_candidate in self.finished_candidates)
                   or (new_candidate in self.working_candidates)
                   or (new_candidate in self.pending_candidates)):
                new_candidate_point = self._generate_new_random_vector()
                new_candidate = Candidate(new_candidate_point)

            logging.debug("Core generated new point to evaluate " +
                          str(new_candidate))

        #add candidate to working list
        self.working_candidates.append(new_candidate)

        return new_candidate

    def _generate_new_random_vector(self):
        """
        Generates a new random vector fitting the self.lower_bound and
        self.upper_bound specifications.
        """
        new_candidate_point = np.zeros(self.lower_bound.shape)

        for i in range(new_candidate_point.shape[0]):
            new_candidate_point[i] = self.random_state.uniform(
                self.lower_bound[i],
                self.upper_bound[i])

        return new_candidate_point

